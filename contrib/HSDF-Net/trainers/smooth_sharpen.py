import os
#import torch
import jittor
import importlib
import os.path as osp
from trainers.base_trainer import BaseTrainer
from models.igp_wrapper import distillation, deformation, correction
from trainers.utils.utils import set_random_seed
from trainers.losses.eikonal_loss import loss_eikonal
from trainers.losses.filtering_losses import loss_boundary, loss_lap


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        self.dim = 3

        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        if original_decoder is None:
            if not hasattr(cfg.models, "net"):
                cfg.models.net = cfg.models.decoder
            sn_lib = importlib.import_module(cfg.models.net.type)
            self.original_decoder = sn_lib.Net(cfg, cfg.models.net)
            self.original_decoder.cuda()
            self.original_decoder.load_state_dict(
                jittor.load(cfg.models.net.path)['net'])
            print("Original Decoder:")
            print(self.original_decoder)
        else:
            self.original_decoder = original_decoder

        # Get the wrapper for the operation
        self.wrapper_type = getattr(
            cfg.trainer, "wrapper_type", "distillation")
        if self.wrapper_type in ['distillation']:
            self.decoder, self.opt_dec, self.scheduler_dec = distillation(
                cfg, self.original_decoder,
                reload=getattr(self.cfg.trainer, "reload_decoder", True))
        elif self.wrapper_type in ['correction']:
            self.decoder, self.opt_dec, self.scheduler_dec = correction(
                cfg, self.original_decoder)
        elif self.wrapper_type in ['deformation']:
            self.decoder, self.opt_dec, self.scheduler_dec = deformation(
                cfg, self.original_decoder)
        else:
            raise ValueError("wrapper_type:", self.wrapper_type)

        # Prepare save directory
        os.makedirs(osp.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "val"), exist_ok=True)

        # Set-up counter
        self.num_update_step = 0
        self.boundary_points = None

        # The [beta] that controlls how smooth/sharp the output shape should be
        # If beta >  1, then the output shape will increase in curvature
        #               so it will be sharper
        # If beta < 1, then the output shape will decrease in curvature
        #               so it will be smoother.
        # beta should be > 0.
        self.beta = getattr(self.cfg.trainer, "beta", 1.)

        # whether plot histogram for network weights
        self.show_network_hist = getattr(
            self.cfg.trainer, "show_network_hist", False)

    def update(self, _, *args, **kwargs):
        self.num_update_step += 1
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.decoder.train()
            self.opt_dec.zero_grad()

        boundary_loss_weight = float(getattr(
            self.cfg.trainer, "boundary_weight", 1.))
        boundary_loss_num_points = int(getattr(
            self.cfg.trainer, "boundary_num_points", 0))
        boundary_loss_points_update_step = int(getattr(
            self.cfg.trainer, "boundary_loss_points_update_step", 1))
        boundary_loss_use_surf_points = int(getattr(
            self.cfg.trainer, "boundary_loss_use_surf_points", True))
        if boundary_loss_weight > 0. and boundary_loss_num_points > 0:
            if self.num_update_step % boundary_loss_points_update_step == 0:
                self.boundary_points = None
            loss_y_boundary, self.boundary_points = loss_boundary(
                (lambda x: self.original_decoder(x)),
                (lambda x: self.decoder(x)),
                npoints=boundary_loss_num_points,
                x=self.boundary_points,
                dim=self.dim,
                use_surf_points=boundary_loss_use_surf_points)
            loss_y_boundary = loss_y_boundary * boundary_loss_weight
        else:
            loss_y_boundary = jittor.zeros(1).float().cuda()

        grad_norm_weight = float(getattr(
            self.cfg.trainer, "grad_norm_weight", 1e-2))
        grad_norm_num_points = int(getattr(
            self.cfg.trainer, "grad_norm_num_points", 5000))
        if grad_norm_weight > 0. and grad_norm_num_points > 0:
            loss_unit_grad_norm = loss_eikonal(
                lambda x: self.decoder(x),
                npoints= grad_norm_num_points,
                use_surf_points=False, invert_sampling=False
            )
            loss_unit_grad_norm *= grad_norm_weight
        else:
            loss_unit_grad_norm = jittor.zeros(1).float().cuda()

        lap_loss_weight = float(getattr(
            self.cfg.trainer, "lap_loss_weight", 1e-4))
        lap_loss_threshold = int(getattr(
            self.cfg.trainer, "lap_loss_threshold", 50))
        lap_loss_num_points = int(getattr(
            self.cfg.trainer, "lap_loss_num_points", 5000))
        if lap_loss_weight > 0. and lap_loss_num_points > 0:
            loss_lap_scaling = loss_lap(
                (lambda x: self.original_decoder(x)),
                (lambda x: self.decoder(x)),
                npoints=lap_loss_num_points,
                beta=self.beta,
                masking_thr=lap_loss_threshold,
            )
            loss_lap_scaling = loss_lap_scaling * lap_loss_weight
        else:
            loss_lap_scaling = jittor.zeros(1).float().cuda()

        loss = loss_unit_grad_norm + loss_y_boundary + loss_lap_scaling
        if not no_update:
            loss.backward()
            self.opt_dec.step()

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss/loss': loss.detach().cpu().item(),
            'scalar/loss/loss_boundary': loss_y_boundary.detach().cpu().item(),
            'scalar/loss/loss_eikonal': loss_unit_grad_norm.detach().cpu().item(),
            'scalar/loss/loss_lap_scaling': loss_lap_scaling.detach().cpu().item(),
            'scalar/weight/loss_boundary': boundary_loss_weight,
            'scalar/weight/loss_eikonal': grad_norm_weight,
            'scalar/weight/loss_lap': lap_loss_weight,
        }

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
            for name, p in self.decoder.named_parameters():
                writer.add_histogram("hist/%s" % name, p, writer_step)

        if visualize:
            # NOTE: trainer sub class should implement this function
            self.visualize(train_info, train_data, writer=writer, step=step,
                           epoch=epoch, visualize=visualize, **kwargs)

    def validate(self, test_loader, epoch, *args, **kwargs):
        return {}

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'orig_dec': self.original_decoder.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'dec': self.decoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = osp.join(self.cfg.save_dir, "checkpoints", save_name)
        jittor.save(d, path)

    def resume(self, path, strict=True, **kwargs):
        ckpt = jittor.load(path)
        self.original_decoder.load_state_dict(ckpt['orig_dec'], strict=strict)
        self.decoder.load_state_dict(ckpt['dec'], strict=strict)
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.scheduler_dec.get_lr()[0], epoch)
