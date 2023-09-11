import os
#import torch
import jittor
import importlib
import os.path as osp
from argparse import Namespace
#import torch.nn.functional as F
import jittor.nn as F
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import set_random_seed
from trainers.utils.igp_utils import sample_points
from trainers.losses.eikonal_loss import loss_eikonal
from models.igp_wrapper import distillation, deformation
from trainers.losses.implicit_thin_shell_losses import \
    stretch_loss, bending_loss


def deform_step(
        net, opt, original, handles_ts, targets_ts, dim=3,
        # Clip gradient
        grad_clip=None,
        # Sample points
        sample_cfg=None, x=None, weights=1,
        # Loss handle
        loss_h_weight=1., use_l1_loss=False, loss_h_thr=None,
        # Loss G
        loss_g_weight=1e-2, n_g_pts=5000,
        # Loss bending
        loss_hess_weight=0., n_hess_pts=5000, hess_use_surf_points=True,
        hess_invert_sample=True, hess_detach_weight=True, hess_use_rejection=False,
        # Loss stretch
        loss_stretch_weight=0., n_s_pts=5000, stretch_use_surf_points=True,
        stretch_invert_sample=True, stretch_loss_type='area_length',
        stretch_use_weight=False, stretch_detach_weight=True,
        stretch_use_rejection=False,
):
    opt.zero_grad()

    # Compute handle losses
    # x
    handles_ts = handles_ts.clone().detach().float().cuda()
    # y
    targets_ts = targets_ts.clone().detach().float().cuda()
    constr = (
            net(targets_ts, return_delta=True)[0] + targets_ts - handles_ts
    ).view(-1, dim).norm(dim=-1, keepdim=False)
    if loss_h_thr is not None:
        loss_h_thr = float(loss_h_thr)
        constr = F.relu(constr - loss_h_thr)
    if use_l1_loss:
        loss_h = F.l1_loss(
            constr, jittor.zeros_like(constr)) * loss_h_weight
    else:
        loss_h = F.mse_loss(
            constr, jittor.zeros_like(constr)) * loss_h_weight

    if sample_cfg is not None and x is None:
        x, weights = sample_points(
            npoints=getattr(sample_cfg, "num_points", 5000),
            dim=dim, inp_nf=original, out_nf=net, deform=net.deform,
            sample_surf_points=getattr(sample_cfg, "use_surf_points", True),
            invert_sampling=getattr(sample_cfg, "invert_sample", True),
            detach_weight=getattr(sample_cfg, "detach_weight", True),
            use_rejection=getattr(sample_cfg, "use_rejection", False)
        )

    if loss_g_weight > 0.:
        loss_g = loss_eikonal(net, npoints=n_g_pts, dim=dim, x=x) * loss_g_weight
    else:
        loss_g = jittor.zeros(1).cuda().float()

    if loss_hess_weight > 0.:
        loss_hess = bending_loss(
            inp_nf=original, out_nf=net, deform=net.deform,
            dim=dim, npoints=n_hess_pts,
            use_surf_points=hess_use_surf_points,
            invert_sampling=hess_invert_sample,
            x=x, weights=weights,
            detach_weight=hess_detach_weight,
            use_rejection=hess_use_rejection,
        )
        loss_hess *= loss_hess_weight
    else:
        loss_hess = jittor.zeros(1).cuda().float()

    if loss_stretch_weight > 0.:
        loss_stretch = stretch_loss(
            inp_nf=original, out_nf=net, deform=net.deform,
            npoints=n_s_pts, dim=dim,
            use_surf_points=stretch_use_surf_points,
            invert_sampling=stretch_invert_sample,
            loss_type=stretch_loss_type,
            x=x, weights=weights,
            detach_weight=stretch_detach_weight,
            use_rejection=stretch_use_rejection,
        )
        loss_stretch *= loss_stretch_weight
    else:
        loss_stretch = jittor.zeros(1).cuda().float()

    loss = loss_h + loss_g + loss_hess + loss_stretch
    opt.backward(loss)
    if grad_clip is not None:
        opt.clip_grad_norm(net.deform.parameters(), grad_clip)

    opt.step()

    return {
        'loss': loss.detach().cpu().item(),
        'loss_h': loss_h.detach().cpu().item(),
        # Repairing
        'loss_g': loss_g.detach().cpu().item(),
        # Shell energy
        'loss_hess': loss_hess.detach().cpu().item(),
        'loss_stretch': loss_stretch.detach().cpu().item()
    }
    

class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        # TODO: add recursive loading of trainers.
        if original_decoder is None:
            sn_lib = importlib.import_module(cfg.models.decoder.type)
            self.original_net = sn_lib.Net(cfg, cfg.models.decoder)
            self.original_net.cuda()
            self.original_net.load_state_dict(
                jittor.load(cfg.models.decoder.path)['net'])
            print("Original Decoder:")
            print(self.original_net)
        else:
            self.original_net = original_decoder

        # Get the wrapper for the operation
        self.wrapper_type = getattr(
            cfg.trainer, "wrapper_type", "distillation")
        if self.wrapper_type in ['distillation']:
            self.net, self.opt, self.sch = distillation(
                cfg, self.original_net,
                reload=getattr(self.cfg.trainer, "reload_decoder", True))
        elif self.wrapper_type in ['deformation']:
            self.net, self.opt, self.sch = deformation(
                cfg, self.original_net)
        else:
            raise ValueError("wrapper_type:", self.wrapper_type)

        # Prepare save directory
        os.makedirs(osp.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "val"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "vis"), exist_ok=True)

        # Set-up counter
        self.num_update_step = 0
        self.boundary_points = None

        # Set up basic parameters
        self.dim = getattr(cfg.trainer, "dim", 3)
        self.grad_clip = getattr(cfg.trainer, "grad_clip", None)
        self.loss_h_weight = getattr(cfg.trainer, "loss_h_weight", 100)
        self.loss_h_thr = getattr(cfg.trainer, "loss_h_thr", 1e-3)

        if hasattr(cfg.trainer, "loss_g"):
            self.loss_g_cfg = cfg.trainer.loss_g
        else:
            self.loss_g_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_bend"):
            self.loss_bend_cfg = cfg.trainer.loss_bend
        else:
            self.loss_bend_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_stretch"):
            self.loss_stretch_cfg = cfg.trainer.loss_stretch
        else:
            self.loss_stretch_cfg = Namespace()

        if hasattr(cfg.trainer, "sample_cfg"):
            self.sample_cfg = cfg.trainer.sample_cfg
        else:
            self.sample_cfg = None

        self.show_network_hist = getattr(
            cfg.trainer, "show_network_hist", False)

    def update(self, data, *args, **kwargs):
        self.num_update_step += 1
        handles_ts = data['handles'].cuda().float()
        targets_ts = data['targets'].cuda().float()
        if 'x' in data and 'weights' in data:
            x_ts = data['x'].cuda().float()
            w_ts = data['weights'].cuda().float()
        else:
            x_ts = None
            w_ts = 1.

        loss_g_weight = float(getattr(self.loss_g_cfg, "weight", 1e-3))
        loss_hess_weight = float(getattr(self.loss_bend_cfg, "weight", 0.))
        loss_stretch_weight = float(
            getattr(self.loss_stretch_cfg, "weight", 0))
        step_res = deform_step(
            self.net, self.opt, self.original_net,
            handles_ts, targets_ts, dim=self.dim,
            x=x_ts, weights=w_ts,
            sample_cfg=self.sample_cfg,
            # Loss handle
            loss_h_weight=self.loss_h_weight,
            loss_h_thr=self.loss_h_thr,
            # Loss G
            loss_g_weight=loss_g_weight,
            n_g_pts=getattr(self.loss_g_cfg, "num_points", 5000),

            # Loss Hessian
            loss_hess_weight=loss_hess_weight,
            n_hess_pts=getattr(self.loss_bend_cfg, "num_points", 5000),
            hess_use_surf_points=getattr(
                self.loss_bend_cfg, "use_surf_points", True),
            hess_invert_sample=getattr(
                self.loss_bend_cfg, "invert_sample", True),
            hess_detach_weight=getattr(
                self.loss_bend_cfg, "detach_weight", True),
            hess_use_rejection=getattr(
                self.loss_bend_cfg, "use_rejection", True),

            # Loss stretch
            loss_stretch_weight=loss_stretch_weight,
            n_s_pts=getattr(self.loss_stretch_cfg, "num_points", 5000),
            stretch_use_surf_points=getattr(
                self.loss_stretch_cfg, "use_surf_points", True),
            stretch_invert_sample=getattr(
                self.loss_stretch_cfg, "invert_sample", True),
            stretch_loss_type=getattr(
                self.loss_stretch_cfg, "loss_type", "l2"),
            stretch_use_weight=getattr(
                self.loss_stretch_cfg, "use_weight", True),
            stretch_detach_weight=getattr(
                self.loss_stretch_cfg, "detach_weight", True),
            stretch_use_rejection=getattr(
                self.loss_stretch_cfg, "use_rejection", True),

            # Gradient clipping
            grad_clip=self.grad_clip,
        )
        step_res = {
            ('scalar/loss/%s' % k): v for k, v in step_res.items()
        }
        step_res['loss'] = step_res['scalar/loss/loss']
        step_res.update({
            "scalar/weight/loss_h_weight": self.loss_h_weight,
            'scalar/weight/loss_hess_weight': loss_hess_weight,
            'scalar/weight/loss_stretch_weight': loss_stretch_weight,
        })
        return step_res

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return
        writer_step = step if step is not None else epoch

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        for k, v in train_info.items():
            ktype = k.split("/")[0]
            kstr = "/".join(k.split("/")[1:])
            if ktype == 'scalar':
                writer.add_scalar(kstr, v, writer_step)

        if self.show_network_hist:
            for name, p in self.net.named_parameters():
                writer.add_histogram("dec/%s" % name, p, writer_step)
            for name, p in self.original_net.named_parameters():
                writer.add_histogram("orig_dec/%s" % name, p, writer_step)

    def validate(self, test_loader, epoch, *args, **kwargs):
        # TODO: compute mesh and compute the manifold harmonics to
        #       see if the high frequencies signals are dimed/suppressed
        return {}

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'dec': self.original_net.state_dict(),
            'net_opt_dec': self.opt.state_dict(),
            'next_dec': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        jittor.save(d, osp.join(self.cfg.save_dir, "checkpoints", save_name))
        jittor.save(d, osp.join(self.cfg.save_dir, "latest.pt"))

    def resume(self, path, strict=True, **kwargs):
        ckpt = jittor.load(path)
        self.original_net.load_state_dict(ckpt['dec'], strict=strict)
        self.net.load_state_dict(ckpt['next_dec'], strict=strict)
        self.opt.load_state_dict(ckpt['net_opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.sch is not None:
            self.sch.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'lr/opt_dec_lr_sch', self.sch.get_lr()[0], epoch)
