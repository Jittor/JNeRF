import os
#import torch
import jittor
import trimesh
import numpy as np
from argparse import Namespace
from trainers.utils.vis_utils import imf2mesh
from evaluation.evaluation_metrics import CD, EMD
from trainers.implicit_deform import Trainer as BaseTrainer
from trainers.utils.igp_utils import compute_invert_weight


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args, original_decoder=original_decoder)
        self.dim = 3
        self.vis_cfg = getattr(self.cfg.trainer, "vis", Namespace())

        # same resolution as the one from
        self.res = int(getattr(self.cfg.trainer, "mc_res", 256))
        self.thr = float(getattr(self.cfg.trainer, "mc_thr", 0.))
        self.original_mesh, self.original_mesh_stats = imf2mesh(
            lambda x: self.original_net(x),
            res=self.res, threshold=self.thr,
            normalize=True, norm_type='res', return_stats=True
        )

        if hasattr(self.cfg.trainer, "mesh_presample"):
            self.presample_cfg = self.cfg.trainer.mesh_presample
            self.presmp_npoints = getattr(
                self.presample_cfg, "num_points", 10000)
        else:
            self.presmp_npoints = None

    def update(self, data, *args, **kwargs):
        if self.presmp_npoints is not None:
            uniform_pcl_orig = self.original_mesh.sample(self.presmp_npoints)
            with jittor.no_grad():
                x_invert_uniform = self.net.deform.invert(
                    jittor.Var(uniform_pcl_orig).float().cuda().view(-1, 3),
                    iters=30
                ).view(1, -1, 3).cuda().float()

            weights = compute_invert_weight(
                x_invert_uniform,
                inp_nf=self.original_net,
                out_nf=self.net,
                deform=self.net.deform,
                surface=True,
            ).cuda().float().view(1, -1)

            if getattr(self.presample_cfg, "detach_weight", False):
                weights = weights.detach()

            data.update({
                'x': x_invert_uniform,
                'weights': weights
            })
        return super().update(data, *args, **kwargs)

    def visualize(
            self, train_data, train_info,
            writer=None, step=None, epoch=None, **kwargs):
        res = int(getattr(self.cfg.trainer, "vis_mc_res", 64))
        thr = float(getattr(self.cfg.trainer, "vis_mc_thr", 0.))
        with jittor.no_grad():
            print("Visualize: %s %s" % (step, epoch))
            mesh = imf2mesh(
                lambda x: self.net(x, None), res=res, threshold=thr,
                normalize=True, norm_type='res'
            )
            if mesh is not None:
                save_name = "mesh_%diters.obj" \
                            % (step if step is not None else epoch)
                path = os.path.join(self.cfg.save_dir, "vis", save_name)
                mesh.export(path)

    def validate(self, test_loader, epoch, *args, **kwargs):
        print("Validating : %d" % epoch)

        cd_gtr = 0
        emd_gtr = 0
        cd_out = 0
        emd_out = 0
        cd_ratio = 0.
        emd_ratio = 0
        area_ratio = 0.
        vol_ratio = 0.

        with jittor.no_grad():
            new_mesh, new_mesh_stats = imf2mesh(
                lambda x: self.net(x),
                res=self.res, threshold=self.thr,
                normalize=True, norm_type='res', return_stats=True
            )
            if new_mesh is not None:
                save_name = "mesh_%diters.obj" % epoch
                path = os.path.join(self.cfg.save_dir, "val", save_name)
                new_mesh.export(path)

                area_ratio = new_mesh_stats['area'] / (self.original_mesh_stats['area'] + 1e-5)
                vol_ratio = new_mesh_stats['vol'] / (self.original_mesh_stats['vol'] + 1e-5)

                for test_data in test_loader:
                    break
                if 'gtr_verts' in test_data and 'gtr_faces' in test_data:
                    npoints = getattr(self.cfg.trainer, "val_npoints", 2048)
                    gtr_verts = test_data['gtr_verts'].detach().view(-1, 3).cpu().numpy()
                    gtr_faces = test_data['gtr_faces'].detach().view(-1, 3).cpu().numpy()
                    gtr_mesh = trimesh.Trimesh(vertices=gtr_verts, faces=gtr_faces)

                    gtr_pcl0 = gtr_mesh.sample(npoints)[np.newaxis, ...]
                    gtr_pcl1 = gtr_mesh.sample(npoints)[np.newaxis, ...]
                    out_pcl = new_mesh.sample(npoints)[np.newaxis, ...]
                    print(gtr_pcl0.shape, gtr_pcl1.shape, out_pcl.shape)

                    cd_gtr, dists_gtr = CD(
                        jittor.Var(gtr_pcl0), jittor.Var(gtr_pcl1))
                    cd_out, dists_out = CD(
                        jittor.Var(gtr_pcl0), jittor.Var(out_pcl))
                    cd_ratio = cd_out / (cd_gtr + 1e-8)

                    emd_gtr, _ = EMD(
                        jittor.Var(gtr_pcl0), jittor.Var(gtr_pcl1),
                        dist=dists_gtr
                    )
                    emd_out, _ = EMD(
                        jittor.Var(gtr_pcl0), jittor.Var(out_pcl),
                        dist=dists_out
                    )
                    emd_ratio = emd_out / (emd_gtr + 1e-8)

        res = {
            'val/org_mesh_area': self.original_mesh_stats['area'],
            'val/org_mesh_vol': self.original_mesh_stats['vol'],
            'val/new_mesh_area': new_mesh_stats['area'],
            'val/new_mesh_vol': new_mesh_stats['vol'],
            'val/area_change_ratio': area_ratio,
            'val/vol_change_ratio': vol_ratio,
            'val/cd_gtr': cd_gtr,
            'val/emd_gtr': emd_gtr,
            'val/cd_out': cd_out,
            'val/emd_out': emd_out,
            'val/cd_ratio': cd_ratio,
            'val/emd_ratio': emd_ratio
        }
        print(res)
        return res


