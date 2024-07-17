import os
#import torch
import jittor
import importlib
import os.path as osp
#import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        lib = importlib.import_module(cfg.models.decoder.type)
        self.net = lib.Net(cfg, cfg.models.decoder)
        self.net.cuda()
        print("Net:")
        print(self.net)

        # The optimizer
        self.opt, self.sch = get_opt(
            self.net.parameters(), self.cfg.trainer.opt)

        # Prepare save directory
        os.makedirs(osp.join(cfg.save_dir, "val"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "checkpoints"), exist_ok=True)

    def update(self, data, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.net.train()
            self.opt.zero_grad()

        xyz, dist = data['xyz'].cuda(), data['dist'].cuda()
        bs = xyz.size(0)
        out = self.net(xyz)
        ndf_loss_weight = float(getattr(
            self.cfg.trainer, "ndf_loss_weight", 1.))
        if ndf_loss_weight > 0:
            loss_y_ndf = ((jittor.abs(out) - dist) ** 2).view(bs, -1).mean()
            loss_y_ndf *= ndf_loss_weight
        else:
            loss_y_ndf = jittor.zeros(1).cuda().float()

        sdf_loss_weight = float(getattr(
            self.cfg.trainer, "sdf_loss_weight", 0.))
        if 'sign' in data and sdf_loss_weight > 0:
            sign = data['sign'].cuda().float()
            loss_y_sdf = ((out - dist * sign) ** 2).view(bs, -1).mean()
            loss_y_sdf *= sdf_loss_weight
        else:
            loss_y_sdf = 0. * jittor.zeros(1).to(loss_y_ndf)

        occ_loss_weight = float(getattr(
            self.cfg.trainer, "occ_loss_weight", 0.))
        if 'sign' in data and occ_loss_weight > 0:
            target = (data['sign'].cuda().float() >= 0).float()
            loss_occ = F.binary_cross_entropy(
                jittor.sigmoid(out), target
            )
            loss_occ *= occ_loss_weight
        else:
            loss_occ = 0. * jittor.zeros(1).cuda().float()

        grad_norm_weight = float(getattr(
            self.cfg.trainer, "grad_norm_weight", 0.))
        grad_norm_num_points = int(getattr(
            self.cfg.trainer, "grad_norm_num_points", 0))
        if grad_norm_weight > 0. and grad_norm_num_points > 0:
            xyz = jittor.rand(
                bs, grad_norm_num_points, xyz.size(-1)).to(xyz) * 2 - 1
            xyz = xyz.cuda()
            xyz.requires_grad = True
            grad_norm = gradient(self.net(xyz), xyz).view(
                bs, -1, xyz.size(-1)).norm(dim=-1)
            loss_unit_grad_norm = F.mse_loss(
                grad_norm, jittor.ones_like(grad_norm)) * grad_norm_weight
        else:
            loss_unit_grad_norm = 0. * jittor.zeros(1).to(loss_y_ndf)
        loss = loss_unit_grad_norm + loss_y_ndf + loss_y_sdf + loss_occ

        if not no_update:
            loss.backward()
            self.opt.step()

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/loss_y_ndf': loss_y_ndf.detach().cpu().item(),
            'scalar/loss_y_sdf': loss_y_sdf.detach().cpu().item(),
            'scalar/loss_occ': loss_occ.detach().cpu().item(),
            'scalar/loss_grad_norm': loss_unit_grad_norm.detach().cpu().item(),
        }

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return

        # Log training information to tensorboard
        writer_step = step if step is not None else epoch
        assert writer_step is not None
        for k, v in train_info.items():
            t, kn = k.split("/")[0], "/".join(k.split("/")[1:])
            if t not in ['scalar']:
                continue
            if t == 'scalar':
                writer.add_scalar('train/' + kn, v, writer_step)

        if visualize:
            with jittor.no_grad():
                print("Visualize: %s" % step)
                res = int(getattr(self.cfg.trainer, "vis_mc_res", 256))
                thr = float(getattr(self.cfg.trainer, "vis_mc_thr", 0.))

                mesh = imf2mesh(
                    lambda x: self.net(x), res=res, threshold=thr)
                if mesh is not None:
                    save_name = "mesh_%diters.obj" \
                                % (step if step is not None else epoch)
                    mesh.export(osp.join(self.cfg.save_dir, "val", save_name))
                    mesh.export(osp.join(self.cfg.save_dir, "latest_mesh.obj"))

    def validate(self, test_loader, epoch, *args, **kwargs):
        return {}

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt': self.opt.state_dict(),
            'net': self.net.state_dict(),
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
        self.net.load_state_dict(ckpt['net'], strict=strict)
        self.opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.sch is not None:
            self.sch.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.sch.get_lr()[0], epoch)
