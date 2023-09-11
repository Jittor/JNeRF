import os
from trainers.utils.vis_utils import imf2mesh
from trainers.smooth_sharpen import Trainer as BaseTrainer


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args, original_decoder=original_decoder)
        self.dim = 3

    def visualize(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        print("Visualize: %s" % step)
        res = int(getattr(self.cfg.trainer, "vis_mc_res", 256))
        thr = float(getattr(self.cfg.trainer, "vis_mc_thr", 0.))
        mesh = imf2mesh(lambda x: self.decoder(x), res=res, threshold=thr)
        if mesh is not None:
            save_name = "mesh_%diters.obj" % self.num_update_step
            mesh.export(os.path.join(self.cfg.save_dir, "val", save_name))
            mesh.export(os.path.join(self.cfg.save_dir, "latest_mesh.obj"))
