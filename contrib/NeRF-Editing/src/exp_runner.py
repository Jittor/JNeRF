import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import jittor as jt
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from tensorboardX import SummaryWriter


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf'])
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'])
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network'])
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'])
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = jt.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

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

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            # near, far = 1 * torch.ones_like(near), 100 * torch.ones_like(far)

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
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            # color_fine_loss = jt.nn.l1_loss(color_error, jt.zeros_like(color_error), reduction='sum') / mask_sum
            color_fine_loss = color_error.abs().sum() / mask_sum
            psnr = 20.0 * jt.log2(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt()) / jt.log2(10)

            eikonal_loss = gradient_error

            mask_loss = jt.nn.binary_cross_entropy_with_logits(weight_sum.safe_clip(1e-3, 1.0 - 1e-3), mask)
            # mask_loss = 0

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            if loss.isnan().any():
                # print("***** Nan loss with %d ray *****" % (color_fine.isnan().sum()))
                # print("Current image index is %d ." % (int(image_perm[self.iter_step % len(image_perm)])))
                continue

            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss.numpy(), self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss.numpy(), self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss.numpy(), self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean().numpy(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', ((cdf_fine[:, :1] * mask).sum() / mask_sum).numpy(), self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', ((weight_max * mask).sum() / mask_sum).numpy(), self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr.numpy(), self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()
                # print("Cancle the image validataion due to CUDA OOM")

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()
                # print("Cancle the mesh validataion due to CUDA OOM")

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return jt.randperm(self.dataset.n_images)
        # logging.debug("Debug by traversing images sequentially")
        # return jt.arange(self.dataset.n_images)

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

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = jt.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
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
                out_rgb_fine.append(render_out['color_fine'].numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

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

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).safe_clip(0, 255).astype(np.uint8)
        return img_fine


    def render_image(self, render_pose, use_deform=False, query_delta=None,
           hull=None, deltas=None, mesh=None, c2w_staticcam=None):
        out_rgb_fine = []
        if mesh != None:
            rays_o, rays_d, depth = self.dataset.gen_rays_at_pose_with_depth(render_pose, mesh, resolution_level=2)
            rays_o, rays_d = self.dataset.gen_rays_at_pose(render_pose, resolution_level=2)
            H, W = depth.shape[:2]
            mask = (depth > 1e-5).reshape(H, W, -1).astype(np.uint8) * 255
            # import imageio
            # imageio.imwrite('./test.jpg', depth.cpu().numpy())
            # import ipdb; ipdb.set_trace()
        else:
            rays_o, rays_d = self.dataset.gen_rays_at_pose(render_pose, resolution_level=2)

        if c2w_staticcam != None:
            view_dirs = rays_d.reshape(-1, 3).split(self.batch_size)
            rays_o, rays_d, depth = self.dataset.gen_rays_at_pose_with_depth(c2w_staticcam, mesh, resolution_level=2)
        else:
            Num = len(rays_o.reshape(-1, 3).split(self.batch_size))
            view_dirs = [None] * Num

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        if mesh != None:
            depth = jt.reshape(depth, [-1,1]).float().split(self.batch_size)  # [H*W, 1]
            epsilon = 0.2
        else:
            depth = [None] * Num

        VIS_RAY = False
        if VIS_RAY:
            w_coord, h_coord = 180, 150  # fox pixel 1
            vis_coord_ind = h_coord * W + w_coord
        else:
            vis_coord_ind = -1

        t1 = time.time()
        for rays_o_batch, rays_d_batch, depth_batch, view_dir in zip(rays_o, rays_d, depth, view_dirs):
            if mesh != None:
                near, far = depth_batch - epsilon, depth_batch + epsilon
            else:
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            background_rgb = jt.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                            rays_d_batch,
                                            near,
                                            far,
                                            view_dir,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                            background_rgb=background_rgb,
                                            use_deform=use_deform,
                                            query_delta=query_delta,
                                            hull=hull,
                                            deltas=deltas, 
                                            vis_coord_ind=vis_coord_ind)
            vis_coord_ind -= self.batch_size

            ttt = time.time()
            # out_rgb_fine.append(render_out['color_fine'].numpy())
            # out_rgb_fine[-1] = np.concatenate([out_rgb_fine[-1], \
            #     render_out['weights'].sum(dim=-1, keepdims=True).numpy()], axis=-1)
            out_rgb_fine.append(np.concatenate([render_out['color_fine'].numpy(), render_out['weights'].sum(dim=-1, keepdims=True).numpy()], axis=-1))
            del render_out
            # print("post process cost %s" % (time.time() - ttt))
            # print("..............")

        print("rendering cost time:", time.time()-t1)
        # img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).safe_clip(0, 255).astype(np.uint8)
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 4]) * 256).clip(0, 255).astype(np.uint8)
        # if mesh != None:
        #     img_fine = np.concatenate([img_fine, mask], axis=-1)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0, with_color=False, do_dilation=False):
        bound_min = jt.float32(self.dataset.object_bbox_min)
        bound_max = jt.float32(self.dataset.object_bbox_max)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, do_dilation=do_dilation)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # if world_space:
        #     vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles, process=False, maintain_order=True)
        no_view_dependence = True

        if with_color:
            normals = jt.array(mesh.vertex_normals.copy()).float()
            normals = -1 * normals / jt.norm(normals, dim=-1, keepdim=True)
            normals = normals.split(self.batch_size)
            pts = jt.array(vertices).float().split(self.batch_size)
            verts_color = []
            if no_view_dependence:
                print("sample according to the vertex position")
                for pts_batch, dir_batch in zip(pts, normals):
                    sdf_nn_output = self.renderer.sdf_network(pts_batch)
                    # sdf = sdf_nn_output[:, :1]
                    feature_vector = sdf_nn_output[:, 1:]  # [bs, 256]
                    gradients = self.renderer.sdf_network.gradient(pts_batch)
                    sampled_color = self.renderer.color_network(pts_batch, gradients, dir_batch, feature_vector)
                    verts_color.append(sampled_color.numpy())
                # del sdf_nn_output, feature_vector, gradients, sampled_color
                verts_color = (np.concatenate(verts_color, axis=0) * 255).clip(0, 255).astype(np.uint8)
            else:
                print("sample along the normal direction")
                rays_o, rays_d = pts, normals
                from tqdm import tqdm
                epsilon = 0.1
                for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
                    near, far = -1 * epsilon * jt.ones_like(rays_o_batch[...,:1]), epsilon * jt.ones_like(rays_o_batch[...,:1])
                    render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio())
                    verts_color.append(render_out['color_fine'].numpy())
                # verts_color = (np.concatenate(verts_color, axis=0) * 255).safe_clip(0, 255).astype(np.uint8)
                verts_color = np.concatenate(verts_color, axis=0)
        # mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
            mesh.visual.vertex_colors = verts_color[:,[2,1,0]]  # modify BGR to RGB
        save_path = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.obj'.format(self.iter_step))
        if do_dilation:
            save_path = save_path.replace('.obj', '_dilation.obj')
        trimesh.exchange.export.export_mesh(mesh, save_path, 'obj')

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
            cv.imwrite(os.path.join(video_dir,
                                    '{:0>8d}.png'.format(i)), images[-1])
            
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


    def render_circle_image(self, recon_file=None, deform_file=None, use_deform=False, obj_path=None, \
                            fix_camera=False, is_view_dependent=False, save_dir="", is_val=False, add_alpha=False):
        if is_val:
            render_poses = self.dataset.gen_validation_pose()
        else:
            render_poses = self.dataset.gen_circle_poses()
        if not save_dir:
            save_dir = os.path.join(self.base_exp_dir, 'render_circle')
        else:
            save_dir = os.path.join(self.base_exp_dir, save_dir)
        if use_deform:
            from utils import genConvexhullVolume, queryDelta
            # from utils import genKNN as genConvexhullVolume
            # from utils import queryDelta_KNN as queryDelta
            hull, deltas = genConvexhullVolume(recon_file, deform_file, fix_camera)
        else:
            hull = deltas = queryDelta = None

        # from pytorch3d.io import load_objs_as_meshes
        from utils import load_objs_as_meshes
        if fix_camera:
            print("FIX CAMERA")
            render_poses = render_poses[0:1].expand(len(deltas), *render_poses.shape[1:])
            ### for laptop
            # render_poses = render_poses[6:7].expand(len(deltas), *render_poses.shape[1:])
            ### for hbychair
            # render_poses = render_poses[26:27].expand(len(deltas), *render_poses.shape[1:])
            ### for dinosaur
            # render_poses = render_poses[22:23].expand(len(deltas), *render_poses.shape[1:])
            import glob
            mesh_files = sorted(glob.glob(os.path.join(args.obj_path, '*.obj')))
            mesh = load_objs_as_meshes(mesh_files)
        else:
            ### load obj file for sampling
            if args.obj_path:
                mesh = load_objs_as_meshes(args.obj_path)
            else:
                mesh = None

        ### copy tets, deltas and meshes
        import copy
        if fix_camera:
            deltas_copy = copy.deepcopy(deltas)
            hull_copy = copy.deepcopy(hull)
            mesh_copy = copy.deepcopy(mesh)

        if is_view_dependent:
            c2w_staticcam = render_poses[0]
        else:
            c2w_staticcam = None
        os.makedirs(save_dir, exist_ok=True)
        images = []
        for idx, render_pose in enumerate(render_poses):
            print("render the %d / %d image" % (idx, len(render_poses)))
            # if idx < 20:
            #     continue
            if fix_camera:
                hull, deltas, mesh = hull_copy[idx], deltas_copy[idx], mesh_copy[idx]
            images.append(self.render_image(render_pose, use_deform, queryDelta, hull, deltas, mesh, c2w_staticcam))
            if add_alpha:
                cv.imwrite(os.path.join(save_dir, '{:0>8d}.png'.format(idx)), images[-1])
            else:
                cv.imwrite(os.path.join(save_dir, '{:0>8d}.png'.format(idx)), images[-1][:,:,:3])

        if images[0].shape[-1] == 4:
            # images = [img[:,:,:3] + (255 - img[:,:,3:]) for img in images]
            images = [img[:,:,:3] for img in images]
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(save_dir, 'video.mp4'),
                                fourcc, 10, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    parser.add_argument("--use_deform", action='store_true', 
                        help='use mesh to guide deformation')
    parser.add_argument("--reconstructed_mesh_file", type=str, default=None, 
                        help='reconstructed mesh path')
    parser.add_argument("--deformed_mesh_file", type=str, default=None, 
                        help='deformed mesh path')

    parser.add_argument("--fix_camera", action='store_true', 
                        help='fix the camera for sequence generation')
    parser.add_argument("--is_view_dependent", action='store_true', 
                        help='fix the camera while change the ray direction')

    # use mesh to for better sampling
    parser.add_argument("--obj_path", type=str, default=None, 
                        help='mesh path')

    # for cage extraction
    parser.add_argument("--do_dilation", action='store_true', 
    help='Optional. Extract cage from current NeRF')

    # for image save
    parser.add_argument("--savedir", type=str, default="", 
                        help='save data directory')

    # use LLFF
    parser.add_argument("--use_llff", action='store_true', 
    help='use llff !')

    # add alpha
    parser.add_argument("--add_alpha", action='store_true', 
    help='add alpha channel')

    args = parser.parse_args()

    if args.mode == 'train' or args.mode == 'default':
        from models.render_train import NeuSRenderer
        print("use train NeuS Renderer: total sampling ...")
    else:
        from models.renderer import NeuSRenderer
        print("use test render NeuS Renderer: sparse sampling ...")

    # if args.use_llff == True:
    #     print("Use colmap extimated poses ~!")
    #     from models.dataset_llff import Dataset
    # else:
    from models.dataset import Dataset

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        if args.do_dilation:
            resol = 256
        else:
            resol = 256
        print("use resoluation %d for marching cube" % resol)
        runner.validate_mesh(world_space=True, resolution=resol, threshold=args.mcube_threshold, \
                            with_color=True, do_dilation=args.do_dilation)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode.startswith('circle'):  # circle views 
        runner.batch_size = 300
        runner.render_circle_image(args.reconstructed_mesh_file, args.deformed_mesh_file,\
                args.use_deform, args.obj_path, args.fix_camera, args.is_view_dependent, args.savedir, add_alpha=args.add_alpha)
    # elif args.mode.startswith('evaluate'):
    #     print("use evaluation poses !!!")
    #     runner.batch_size = 400
    #     runner.render_circle_image(args.reconstructed_mesh_file, args.deformed_mesh_file,\
    #             args.use_deform, args.obj_path, args.fix_camera, args.is_view_dependent, args.savedir, is_val=True)
