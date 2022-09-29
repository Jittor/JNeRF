from jnerf.utils.common import volume_render
import jittor as jt
import jittor.nn as nn
import numpy as np
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, SCHEDULERS, DATASETS, OPTIMS, SAMPLERS, LOSSES

from jnerf.models.samplers.uniform_sampler import UniformSampler
import warnings
warnings.simplefilter("module", UserWarning)


@SAMPLERS.register_module()
class CoarseToFineSampler(nn.Module):
    def __init__(self, perturb=0, N_samples=64,  N_importance=0, use_disp=False, **kwargs) -> None:
        super(CoarseToFineSampler, self).__init__()
        self.cfg = get_cfg()
        self.cfg.coarse_model_obj = build_from_cfg(self.cfg.sampler.coarse_model, NETWORKS)
        self.coarse_model = self.cfg.coarse_model_obj
        self.perturb = perturb  # factor to perturb the sampling position on the ray
        self.N_samples = N_samples  # number of coarse samples per ray
        self.use_disp = use_disp  # whether to sample in disparity space (inverse depth)
        self.N_importance = N_importance  # number of fine samples per ray
        self.uniform_sampler = UniformSampler(perturb=perturb, N_samples=N_samples, use_disp=use_disp)
        self.using_fp16 = self.cfg.using_fp16
        self.background_color = jt.ones([3]).stop_grad()

    def sample(self, img_ids, rays_o, rays_d, rgb_target=None, is_training=False):
        if len(rays_d.shape)==3 and rays_d.shape[2]==1:
            rays_d=rays_d.squeeze(-1)
        self.dirs = rays_d # save for rendering
        self.N_rays = rays_d.shape[0]
        # breakpoint()
        xyz_coarse_sampled, z_vals = self.uniform_sampler.sample(img_ids=img_ids, rays_o=rays_o, rays_d=rays_d, with_z=True,is_training=is_training) # uniform sample
        viewdirs = rays_d/jt.norm(rays_d, p=2, dim=-1, keepdim=True)
        # viewdirs = jt.reshape(viewdirs, [-1,3]).float()
        print("xyz: ", xyz_coarse_sampled.shape)
        enc_dir = self.dataset.feature_matching(xyz_coarse_sampled, rays_o, img_ids)
        rgbsigma_coarse = self.coarse_model(enc_dir, xyz_coarse_sampled, viewdirs)
        # print('coarse model',rgbsigma_coarse.numpy().sum(),xyz_coarse_sampled.numpy().sum(),viewdirs.numpy().sum())
        rgb_coarse = rgbsigma_coarse[..., :3].reshape(-1, self.N_samples, 3)
        sigma_coarse = rgbsigma_coarse[..., 3].reshape(-1, self.N_samples)
        # noise = jt.randn(sigma_coarse.shape)
        noise = 0.0
        rgbs_coarse, depths_coarse, weights_coarse = volume_render(sigmas=sigma_coarse+noise, z_vals=z_vals, raw_rgbs=rgb_coarse, dirs=rays_d)
        rgbs_coarse_final = rgbs_coarse + 1. - weights_coarse.sum(-1).unsqueeze(-1)
        self.coarse_rgbs = rgbs_coarse_final
        # breakpoint()
        if self.N_importance > 0:
            # sample points for fine model
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
            z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1], self.N_importance, det=(self.perturb == 0)or(not is_training)).detach()
            _, z_vals = jt.argsort(jt.concat([z_vals, z_vals_], -1), -1)
            xyz_fine_sampled = rays_o.unsqueeze(1)+rays_d.unsqueeze(1)*z_vals.unsqueeze(2)
            self.z_vals = z_vals # save for rendering
            print("fine: ", xyz_fine_sampled.shape)
            return xyz_fine_sampled, viewdirs, enc_dir
        else:
            warnings.warn("N_importance value is zero, coarse to fine sampler will return coarse sampled points")
            self.z_vals = z_vals
            print("coarse: ", xyz_coarse_sampled.shape)
            return xyz_coarse_sampled, viewdirs, enc_dir

    def rays2rgb(self, network_outputs, training_background_color=None, inference=False):
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.rays2rgb_(network_outputs, training_background_color, inference)
        else:
            return self.rays2rgb_(network_outputs, training_background_color, inference)

    def rays2rgb_(self, network_outputs, training_background_color=None, inference=False):
        if training_background_color is None:
            background_color = self.background_color
        else:
            background_color = training_background_color
        sigmas = network_outputs[..., 3].reshape(self.N_rays, -1)
        raw_rgbs = network_outputs[..., :3].reshape(self.N_rays, -1, 3)
        rgbs, depths, weights = volume_render(sigmas, self.z_vals, raw_rgbs, self.dirs)
        acc = jt.sum(weights, -1)
        # breakpoint()
        alpha = 1. - acc.unsqueeze(-1)
        
        if inference:
            return rgbs,1-alpha
        else:
            rgb_final = rgbs + alpha
            return rgb_final


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    '''
    Input:

    Output:
    '''
    # breakpoint()
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[:, :1]), cdf], -1)  # TODO:

    if det:
        u = jt.linspace(0, 1, N_importance)
        u = u.expand(N_rays, N_importance)
    else:
        u = jt.rand(N_rays, N_importance)

    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.clamp(inds-1, min_v=0)
    above = jt.clamp(inds, max_v=N_samples_)

    inds_sampled = jt.stack([below, above], -1).view(N_rays, N_importance*2)
    cdf_g = jt.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = jt.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)
    denom = cdf_g[..., 1]-cdf_g[..., 0]
    denom[denom < eps] = 1

    samples = bins_g[..., 0]+(u - cdf_g[..., 0])/denom*(bins_g[..., 1]-bins_g[..., 0])
    return samples

