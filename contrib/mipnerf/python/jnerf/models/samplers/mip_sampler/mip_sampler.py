import os
from functools import partial
import jittor as jt
from jittor import Function, nn
import numpy as np
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import SAMPLERS
from jnerf.utils.miputils import *


@SAMPLERS.register_module()
class MipSampler(jt.Function):
    def __init__(self, update_den_freq=16):
        self.cfg = get_cfg()
        self.disable_integration = self.cfg.disable_integration
        self.use_viewdirs = self.cfg.use_viewdirs
        self.min_deg_point = self.cfg.min_deg_point
        self.max_deg_point = self.cfg.max_deg_point
        self.using_fp16 = self.cfg.using_fp16
        self.resample_padding = self.cfg.resample_padding
        self.num_samples = self.cfg.num_samples
        self.randomized = self.cfg.randomized
        self.lindisp = self.cfg.lindisp
        self.ray_shape = self.cfg.ray_shape
        self.stop_level_grad = self.cfg.stop_level_grad
        self.deg_view = self.cfg.deg_view
        self.white_bkgd = self.cfg.white_bkgd
        self.density_noise = self.cfg.density_noise
        self.rgb_activation = partial(nn.Sigmoid())
        self.density_activation = partial(nn.Softplus())
        self.rgb_padding = self.cfg.rgb_padding
        self.density_bias = self.cfg.density_bias

    def sample(self, rays, i_level, t_vals=None, weights=None):
        if i_level == 0:
            # Stratified sampling along rays
            t_vals, samples = sample_along_rays(
                rays.origins,
                rays.directions,
                rays.radii,
                self.num_samples,
                rays.near,
                rays.far,
                self.randomized,
                self.lindisp,
                self.ray_shape,
            )
        else:
            t_vals, samples = resample_along_rays(
                rays.origins,
                rays.directions,
                rays.radii,
                t_vals,
                weights,
                self.randomized,
                self.ray_shape,
                self.stop_level_grad,
                resample_padding=self.resample_padding,
            )
        if self.disable_integration:
            samples = (samples[0], jt.zeros_like(samples[1]))
        samples_enc = integrated_pos_enc(
            samples,
            self.min_deg_point,
            self.max_deg_point,
        )
        # jt.sync_all()
        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_enc = pos_enc(
                rays.viewdirs,
                min_deg=0,
                max_deg=self.deg_view,
                append_identity=True,
                using_fp16=self.using_fp16
            )
        else:
            viewdirs_enc = None
        if self.using_fp16:
            viewdirs_enc = viewdirs_enc.float16()
        return samples_enc, viewdirs_enc, t_vals

    def rays2rgb(self, rays, raw_rgb, raw_density, t_vals):
        # Add noise to regularize the density predictions if needed.
        if self.randomized and (self.density_noise > 0):
            raw_density += self.density_noise * jt.array(np.random.normal(0, 1, raw_density.shape))
        # Volumetric rendering.
        rgb = self.rgb_activation(raw_rgb)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        density = self.density_activation(raw_density + self.density_bias)
        return volumetric_rendering(
            rgb,
            density,
            t_vals,
            rays.directions,
            white_bkgd=self.white_bkgd)
