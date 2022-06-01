import jittor as jt
from jittor import nn
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import ENCODERS
from .grid_encode import GridEncode

@ENCODERS.register_module()
class HashEncoder(nn.Module):
    def __init__(self, n_pos_dims=3, n_features_per_level=2, n_levels=16, base_resolution=16, log2_hashmap_size=19):
        self.cfg = get_cfg()
        using_fp16 = self.cfg.fp16
        aabb_scale = self.cfg.dataset_obj.aabb_scale
        self.hash_func = self.cfg.hash_func
        self.hash_func_header = f"""
#define get_index(p0,p1,p2) {self.hash_func}
        """
        self.encoder = GridEncode(self.hash_func_header, aabb_scale=aabb_scale, n_pos_dims=3, n_features_per_level=2,
                                  n_levels=16, base_resolution=16, log2_hashmap_size=19, using_fp16=using_fp16)
        self.grad_type = 'float32'
        if using_fp16:
            self.grad_type = 'float16'
        self.m_grid = jt.init.uniform(
            [self.encoder.m_n_params], low=-1e-4, high=1e-4, dtype=self.grad_type)
        self.out_dim=n_features_per_level*n_levels

    def execute(self, x):
        assert(self.m_grid.dtype == self.grad_type)
        output = self.encoder(x, self.m_grid)
        assert(output.dtype == self.grad_type)
        return output
