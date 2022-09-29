import os
import jittor as jt
from jittor import nn
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import SAMPLERS
from jnerf.ops.code_ops.global_vars import global_headers, proj_options
from math import ceil, log2

@SAMPLERS.register_module()
class UniformSampler(nn.Module):
    '''
    Uniform sample on ray
    '''
    def __init__(self, perturb=0, N_samples=64, use_disp=False):
        super(UniformSampler, self).__init__()
        self.cfg = get_cfg()
        self.perturb = perturb  # factor to perturb the sampling position on the ray
        self.N_samples = N_samples  # number of samples per ray
        self.near = self.cfg.near
        self.far = self.cfg.far
        assert self.near is not None and self.far is not None,"near and far need to be set"
        self.use_disp = use_disp

    def sample(self, img_ids, rays_o, rays_d, with_z=False, is_training=False,**kwargs):
        '''
        Input:
        rays_o : (N_rays,3)
        rays_d : (N_rays,3)

        '''
        print(rays_o.shape, rays_d.shape)
        N_rays = rays_o.shape[0]
        z_steps = jt.linspace(0, 1, self.N_samples)
        if not self.use_disp:
            z_vals = self.near*(1-z_steps)+self.far*z_steps
        else:
            z_vals = 1/(1/self.near * (1-z_steps) + 1/self.far * z_steps)

        z_vals = z_vals.expand(N_rays, self.N_samples)

        if self.perturb > 0 and is_training:
            z_vals_mid = 0.5*(z_vals[:, :-1]+z_vals[:, 1:])
            upper = jt.concat([z_vals_mid, z_vals[:, -1:]], -1)
            lower = jt.concat([z_vals[:, :1], z_vals_mid], -1)
            perturb_rand = self.perturb*jt.rand(z_vals.shape)
            z_vals = lower + (upper-lower)*perturb_rand
        xyz_sampled = rays_o.unsqueeze(1)+rays_d.unsqueeze(1)*z_vals.unsqueeze(2)
        if with_z:
            return xyz_sampled, z_vals
        else:
            return xyz_sampled
