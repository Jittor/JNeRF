import jittor as jt
from jnerf.utils.registry import DATASETS
from .dataset_util import *

@DATASETS.register_module()
class ReferenceDataset():
    def __init__(self, reference, c2w, focal, img_size):
        self.reference = reference
        self.scale = (img_size / 2) / focal
        self.n = c2w.shape[0]
        self.R_t = jt.array(c2w[:, :3, :3]).permute(0, 2, 1)
        self.camera_pos = jt.array(c2w[:, :3, -1])
        self.c2w = c2w
        self.img_size = img_size
        self.focal = focal

    @jt.no_grad()
    def feature_matching(self, pos):
        n_rays, n_samples, _ = pos.shape
        pos = pos.unsqueeze(dim=0).expand([self.n, n_rays, n_samples, 3])
        camera_pos = self.camera_pos[:, None, None, :]
        camera_pos = camera_pos.expand_as(pos)
        ref_pos = jt.linalg.einsum("kij,kbsj->kbsi", self.R_t, pos-camera_pos)
        uv_pos = ref_pos[..., :-1] / ref_pos[..., -1:] / self.scale
        uv_pos[..., 1] *= -1.0
        return jt.nn.grid_sample(self.reference, uv_pos, align_corners=True, padding_mode="border")
