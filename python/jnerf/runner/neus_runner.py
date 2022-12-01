import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh

from shutil import copyfile
from tqdm import tqdm
from jnerf.dataset.neus_dataset import NeuSDataset
from jnerf.models.networks.neus_network import NeuS
from jnerf.models.samplers.neus_render.renderer import NeuSRenderer

from jnerf.utils.config import init_cfg, get_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES

import jittor as jt
import jittor.nn as nn

class NeuSRunner:
    def __init__(self, mode='train', is_continue=False):

        # Configuration
        self.cfg = get_cfg()

        # basic
        self.base_exp_dir = self.cfg.base_exp_dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.cfg.end_iter
        self.save_freq = self.cfg.save_freq
        self.report_freq = self.cfg.report_freq
        self.val_freq = self.cfg.val_freq
        self.val_mesh_freq = self.cfg.val_mesh_freq
        self.batch_size = self.cfg.batch_size
        self.validate_resolution_level = self.cfg.validate_resolution_level
        self.learning_rate = self.cfg.learning_rate
        self.learning_rate_alpha = self.cfg.learning_rate_alpha
        self.use_white_bkgd = self.cfg.use_white_bkgd
        self.warm_up_end = self.cfg.warm_up_end
        self.anneal_end = self.cfg.anneal_end

        # Weights
        self.igr_weight = self.cfg.igr_weight
        self.mask_weight = self.cfg.mask_weight
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        self.dataset      = build_from_cfg(self.cfg.dataset, DATASETS)
        self.neus_network = build_from_cfg(self.cfg.model, NETWORKS)
        self.renderer     = build_from_cfg(self.cfg.render, SAMPLERS)
        self.renderer.set_neus_network(self.neus_network)

        self.learning_rate = self.cfg.optim.lr
        self.optimizer = build_from_cfg(self.cfg.optim, OPTIMS, params=self.neus_network.parameters())

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pkl' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def train(self):
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = jt.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = jt.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = color_error.abs().sum() / mask_sum

            eikonal_loss = gradient_error

            mask_loss = jt.nn.binary_cross_entropy_with_logits(weight_sum.safe_clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                eikonal_loss * self.igr_weight +\
                mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()

            self.iter_step += 1

            if self.iter_step % self.report_freq == 0 :
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return jt.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def load_checkpoint(self, checkpoint_name):
        checkpoint = jt.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.neus_network.load_state_dict(checkpoint['neus'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'neus': self.neus_network.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        jt.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pkl'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jt.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().numpy())
            
            if feasible('gradients') and feasible('weights') and feasible('z_vals'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                depths  = render_out['z_vals'] * (render_out['weights'][:, :n_samples])

                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                    depths  = depths * render_out['inside_sphere']

                normals = normals.sum(dim=1).detach().numpy()
                depths  = depths.sum(dim=1).detach().numpy()

                out_normal_fine.append(normals)
                out_depth_fine.append(depths)

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        depth_fine = None
        if len(out_depth_fine) > 0:
            depth_fine = np.concatenate(out_depth_fine, axis=0).reshape([H, W])
            depth_fine = cv.applyColorMap(( depth_fine * 255 ).astype(np.uint8),cv.COLORMAP_JET)
            depth_fine = depth_fine.reshape([H,W,3,1])

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            
            depth_fine = np.concatenate(out_depth_fine, axis=0).reshape([H, W])
            depth_fine = cv.applyColorMap(( depth_fine * 255 ).astype(np.uint8),cv.COLORMAP_JET)
            depth_fine = depth_fine.reshape([H,W,3,1])

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'depths'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

            if len(out_depth_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'depths',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           depth_fine[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jt.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):

        bound_min = jt.float32(self.dataset.object_bbox_min)
        bound_max = jt.float32(self.dataset.object_bbox_max)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')
