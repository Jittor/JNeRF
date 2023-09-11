#import torch
import jittor
import numpy as np
from trainers.utils.vis_utils import imf2img, make_2d_grid
from trainers.utils.igp_losses import get_surf_pcl
from trainers.implicit_deform import Trainer as BaseTrainer
from argparse import Namespace
import matplotlib.pyplot as plt


try:
    from evaluation.evaluation_metrics import EMD_CD
    eval_reconstruciton = True
except:  # noqa
    # Skip evaluation
    eval_reconstruciton = False


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args, original_decoder=original_decoder)
        self.dim = 2
        self.vis_cfg = getattr(self.cfg.trainer, "vis", Namespace())

    def visualize(
            self, train_data, train_info,
            writer=None, step=None, epoch=None, **kwargs):
        figsize = getattr(self.vis_cfg, "figsize", 5)
        res = getattr(self.vis_cfg, "res", 100)
        handle_np = (
                (train_data['handles'] + 1.) * float(res) * 0.5
        ).detach().cpu().numpy().reshape(-1, 2)
        new_handle_np = (
                (train_data['targets'] + 1) * float(res) * 0.5
        ).detach().cpu().numpy().reshape(-1, 2)
        orig_img = imf2img(
            lambda x: self.original_net(x, None), res=res
        ).reshape(res, res)
        img = imf2img(
            lambda x: self.net(x, None), res=res).reshape(res, res)

        qres = getattr(self.vis_cfg, "qres", 15)
        orig_loc = make_2d_grid(qres).view(qres * qres, 2).detach().numpy()
        dfm= self.net(
            jittor.Var(orig_loc).cuda().view(1, -1, 2).float(),
            None, return_delta=True
        )[0].detach().cpu().numpy().reshape(qres * qres, 2)

        fig_s = plt.figure(figsize=(figsize, figsize))
        plt.tight_layout()
        axs = fig_s.add_subplot(111)
        axs.contour(orig_img, levels=[0], linestyles='dotted')
        axs.contour(img, levels=[0])
        axs.scatter(handle_np[:, 0], handle_np[:, 1], c='b', marker='*')
        axs.scatter(new_handle_np[:, 0], new_handle_np[:, 1], c='r', marker='o')
        axs.set_title("Shape")
        writer.add_figure(
            "shape", fig_s, global_step=(step if step is not None else epoch))

        fig_f = plt.figure(figsize=(figsize, figsize))
        plt.tight_layout()
        axf = fig_f.add_subplot(111)
        axf.contour(orig_img, levels=[0], linestyles='dotted')
        axf.contour(img)
        axf.scatter(handle_np[:, 0], handle_np[:, 1], c='b', marker='*')
        axf.scatter(new_handle_np[:, 0], new_handle_np[:, 1], c='r', marker='o')
        axf.set_title("Field")
        writer.add_figure(
            "field", fig_f, global_step=(step if step is not None else epoch))

        fig_d = plt.figure(figsize=(figsize, figsize))
        plt.tight_layout()
        axd = fig_d.add_subplot(111)
        axd.set_title("Deform Direction")
        dfm_norm = np.linalg.norm(dfm, axis=-1).reshape(-1, 1)
        dfm_dirc = (dfm / dfm_norm).reshape(-1, 2)
        loc_in_res = 0.5 * (orig_loc + 1.) * res
        axd.contour(orig_img, levels=[0], linestyles='dotted', colors='r')
        axd.contour(img, levels=[0], colors='r')
        axd.quiver(
            loc_in_res[:, 0], loc_in_res[:, 1], dfm_dirc[:, 0], dfm_dirc[:, 1])
        axd.scatter(handle_np[:, 0], handle_np[:, 1], c='b', marker='*')
        axd.scatter(new_handle_np[:, 0], new_handle_np[:, 1], c='r', marker='o')
        writer.add_figure(
            "deform_direction", fig_d,
            global_step=(step if step is not None else epoch))

        fig_l = plt.figure(figsize=(figsize, figsize))
        plt.tight_layout()
        axl = fig_l.add_subplot(111)
        dfm = imf2img(
            lambda x: self.net(x, None, return_delta=True)[0], res=res)
        dfm_norm_img = np.linalg.norm(
            dfm.reshape(res, res, 2), axis=-1).reshape(res, res)
        axl.contourf(dfm_norm_img)
        axl.set_title("Deform Length")
        axl.contour(orig_img, levels=[0], linestyles='dotted', colors='r')
        axl.contour(img, levels=[0], colors='r')
        axl.scatter(handle_np[:, 0], handle_np[:, 1], c='b', marker='*')
        axl.scatter(new_handle_np[:, 0], new_handle_np[:, 1], c='r', marker='o')
        writer.add_figure(
            "deform_length", fig_l,
            global_step=(step if step is not None else epoch))

        # Sample points just to make sure this is also exampled
        # 1. Sample directly with net
        # 2. Sample directly with original
        # 3. Inverse the originally sampled
        n_pts_smp = getattr(self.vis_cfg, "n_pts_smp", 1000)
        print("Sampling forward!")
        x_forward = get_surf_pcl(
            lambda x: self.net(x, None), npoints=n_pts_smp, dim=2
        ).view(-1, 2).detach().cpu().numpy()
        fig_xfwd = plt.figure(figsize=(figsize, figsize))
        plt.tight_layout()
        ax_xfwd = fig_xfwd.add_subplot(111)
        ax_xfwd.set_title("Npoints:%d" % int(x_forward.shape[0]))
        if x_forward.shape[0] > 0:
            ax_xfwd.scatter(x_forward[:, 0], x_forward[:, 1], s=5, marker='o')
        ax_xfwd.set_xlim(-1, 1)
        ax_xfwd.set_ylim(-1, 1)
        writer.add_figure(
            "x_forward", fig_xfwd,
            global_step=(step if step is not None else epoch))

        print("Sampling!")
        x_orig = get_surf_pcl(
            lambda x: self.original_net(x, None),
            npoints=n_pts_smp, dim=2).view(-1, 2)
        if hasattr(self.net, "deform") and x_orig.size(0) > 0 and \
                hasattr(self.net.deform, "invert"):
            with jittor.no_grad():
                x_invert = self.net.deform.invert(
                    x_orig.view(1, -1, 2), iters=30)
                x_invert = x_invert.detach().cpu().numpy().reshape(-1, 2)

                fig_xinv = plt.figure(figsize=(figsize, figsize))
                plt.tight_layout()
                ax_xinv = fig_xinv.add_subplot(111)
                ax_xinv.set_title("Npoints:%d" % int(x_invert.shape[0]))
                ax_xinv.scatter(x_invert[:, 0], x_invert[:, 1], s=5, marker='o')
                ax_xinv.set_xlim(-1, 1)
                ax_xinv.set_ylim(-1, 1)
                writer.add_figure(
                    "x_invert", fig_xinv,
                    global_step=(step if step is not None else epoch))

        x_orig = x_orig.detach().cpu().numpy().reshape(-1, 2)
        fig_xorg = plt.figure(figsize=(figsize, figsize))
        plt.tight_layout()
        ax_xorig = fig_xorg.add_subplot(111)
        ax_xorig.set_title("Npoints:%d" % int(x_orig.shape[0]))
        if x_orig.shape[0] > 0:
            ax_xorig.scatter(x_orig[:, 0], x_orig[:, 1], s=5, marker='o')
        ax_xorig.set_xlim(-1, 1)
        ax_xorig.set_ylim(-1, 1)
        writer.add_figure(
            "x_orig", fig_xorg,
            global_step=(step if step is not None else epoch))


    def validate(self, test_loader, epoch, *args, **kwargs):
        # TODO: compute mesh and compute the manifold harmonics to
        #       see if the high frequencies signals are dimed/suppressed
        val_res = getattr(self.cfg.trainer, "val_res", 128)
        _, orig_stats = imf2img(
            lambda x: self.original_net(x, None), res=val_res,
            return_stats=True, verbose=True
        )
        _, net_stats = imf2img(
            lambda x: self.net(x, None), res=val_res,
            return_stats=True, verbose=True
        )

        return {
            'val/org_area': orig_stats['area'],
            'val/new_area': net_stats['area'],
            'val/area_change_ratio': net_stats['area'] / (orig_stats['area'] + 1e-5),
            'val/org_length': orig_stats['len'],
            'val/new_length': net_stats['len'],
            'val/length_change_ratio': net_stats['len'] / (orig_stats['len'] + 1e-5),
        }

