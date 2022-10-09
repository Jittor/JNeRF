import random
import time

import jittor as jt
from PIL import Image
from jittor.dataset import Dataset
import os
import json
import cv2
import imageio
from math import pi
from math import tan
from tqdm import tqdm
import numpy as np
from jnerf.utils.registry import DATASETS
from jnerf.utils.miputils import Rays, namedtuple_map
from .dataset_util import *

NERF_SCALE = 0.33


@DATASETS.register_module()
class Blender:
    def __init__(self, root_dir, batch_size, mode='train', H=0, W=0, near=0., far=1., img_alpha=True, have_img=True,
                 preload_shuffle=True):
        self.resolution = None
        self.shuffle_index = None
        self.rays = None
        self.img_ids = None
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.preload_shuffle = preload_shuffle
        self.H = H
        self.W = W
        self.scale = 0
        self.transforms_gpu = []  # 读取transform_matrix
        self.metadata = []
        self.image_data = []
        self.focal = None
        self.n_images = 0
        self.img_alpha = img_alpha  # img RGBA or RGB
        assert mode == "train" or mode == "val" or mode == "test"
        self.mode = mode
        self.have_img = have_img
        self.idx_now = 0
        self.near = near
        self.far = far
        self.n_examples = self.n_images
        self.load_data()  # 初始化加载数据
        jt.gc()

    def __next__(self):
        if self.idx_now + self.batch_size >= self.rays.origins.shape[0]:  # bs * iter 大于 n_images * h * w
            rand_idx = jt.randperm(self.rays.origins.shape[0])
            self.img_ids = self.img_ids[rand_idx]
            self.rays = namedtuple_map(lambda r: r[rand_idx], self.rays)
            self.image_data = self.image_data[rand_idx]
            self.idx_now = 0
        img_ids = self.img_ids[self.idx_now:self.idx_now + self.batch_size, 0].int()
        rays = namedtuple_map(lambda r: jt.array(r[self.idx_now:self.idx_now + self.batch_size]), self.rays)
        rgb_target = self.image_data[self.idx_now:self.idx_now + self.batch_size]
        self.idx_now += self.batch_size
        return rays, rgb_target

    def _flatten(self, x):
        # 对gen_rays的rays做展平处理，因为外面包的是list
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        x = np.concatenate(x, axis=0)
        # np.reshape(np.array(x), x[0].shape[-1])
        return x

    def load_data(self, root_dir=None):
        print(f"load {self.mode} data")
        if root_dir is None:
            root_dir = self.root_dir
        # get json file
        json_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1] == ".json":
                    if self.mode in os.path.splitext(file)[0] or (
                            self.mode == "train" and "val" in os.path.splitext(file)[0]):
                        json_paths.append(os.path.join(root, file))
        json_data = None
        # get frames
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if json_data is None:
                json_data = data
            else:
                json_data['frames'] += data['frames']

        frames = json_data['frames']
        if self.mode == "val":
            frames = frames[::10]
        if self.mode == "test":
            frames = frames[::10]
        for frame in tqdm(frames):
            if self.have_img:
                img_path = os.path.join(self.root_dir, frame['file_path'][2:])  # 切片，去掉“ ./ "
                if not os.path.exists(img_path):
                    img_path = img_path + ".png"
                    if not os.path.exists(img_path):
                        continue
                img = read_image(img_path)
                if self.H == 0 or self.W == 0:  # 选第一次的H W
                    self.H = img.shape[0]
                    self.W = img.shape[1]
                self.image_data.append(img)
            else:
                self.image_data.append(np.zeros((self.H, self.W, 3)))
            self.n_images += 1
            matrix = np.array(frame['transform_matrix'], np.float32)  # 这里是读取cmas
            self.transforms_gpu.append(matrix)

        def read_focal_length(resolution: int, axis: str):
            if 'fl_' + axis in json_data:
                return json_data['fl_' + axis]
            elif 'camera_angle_' + axis in json_data:
                return fov_to_focal_length(resolution, json_data['camera_angle_' + axis] * 180 / pi)
            else:
                return 0

        self.H = int(self.H)
        self.W = int(self.W)
        self.resolution = [self.W, self.H]
        self.focal = read_focal_length(self.W, 'x')
        self.image_data = jt.array(self.image_data)
        if self.img_alpha and self.image_data.shape[-1] == 3:
            self.image_data = jt.concat([self.image_data, jt.ones(self.image_data.shape[:-1] + (1,))], -1).stop_grad()
        if self.preload_shuffle:
            self._generate_rays()
            if self.n_images > 1:
                self.img_ids = jt.linspace(0, self.n_images - 1, self.n_images).unsqueeze(-1).repeat(
                    self.H * self.W).reshape(self.n_images * self.H * self.W, -1)
            else:
                self.img_ids = jt.array([0]).unsqueeze(-1).repeat(self.H * self.W).reshape(
                    self.n_images * self.H * self.W, -1)
            self.image_data = self.image_data.reshape(self.n_images * self.H * self.W, -1)
            self.rays = namedtuple_map(self._flatten, self.rays)  # 展平
            t3 = time.time()
            rand_idx = jt.randperm(self.rays.origins.shape[0])  # 打乱次序
            self.img_ids = self.img_ids[rand_idx]
            self.rays = namedtuple_map(lambda r: r[rand_idx], self.rays)
            self.image_data = jt.array(self.image_data[rand_idx])
            t4 = time.time()
            print(t4 - t3)
        jt.gc()

    # TODO(bydeng): Swap this function with a more flexible camera model.
    def _generate_rays_jt(self):
        t1 = time.time()
        """Generating rays for all images."""
        x, y = jt.array(np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.W, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.H, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy'))

        camera_dirs = jt.stack(
            [(x - self.W * 0.5 + 0.5) / self.focal_lengths[0][0],
             -(y - self.H * 0.5 + 0.5) / self.focal_lengths[0][1], -jt.ones_like(x)],
            -1)
        directions = ((camera_dirs[None, ..., None, :] *
                       self.transforms_gpu[:, None, None, :3, :3]).sum(-1))

        origins = self.transforms_gpu[:, None, None, :3, -1].broadcast(
            directions.shape)

        viewdirs = directions / jt.norm(directions, dim=-1, keepdim=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = jt.sqrt(
            jt.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = jt.concat([dx, dx[:, -2:-1, :]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / jt.sqrt(12)  # 这个是像素中心点，映射成像素大小

        ones = jt.ones_like(origins[..., :1])  # 这里是为了匹配rays的长度。都变成N_images * H * W
        self.rays = Rays(
            origins=origins.numpy(),
            directions=directions.numpy(),
            viewdirs=viewdirs.numpy(),
            radii=radii.numpy(),
            lossmult=ones.numpy(),
            near=(ones * self.near).numpy(),
            far=(ones * self.far).numpy())
        t2 = time.time()
        print(t2 - t1)

    def _generate_rays(self):
        """Generating rays for all images."""
        self.w = self.W
        self.h = self.H
        t1 = time.time()
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
            axis=-1)

        directions = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in self.transforms_gpu]

        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(directions, self.transforms_gpu)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(m):
            return [
                m * np.ones_like(origins[i][..., :1])
                for i in range(self.n_images)
            ]

        lossmults = broadcast_scalar_attribute(1).copy()
        nears = broadcast_scalar_attribute(self.near).copy()
        fars = broadcast_scalar_attribute(self.far).copy()

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmults,
            near=nears,
            far=fars)
        del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs
        t2 = time.time()
        print("生成光线时间", t2 - t1)

    def generate_rays_total_test(self, img_ids, H, W):
        """Generating rays for all images.
        这里创建的ray里面的各项，都是list
        """
        self.w = self.W
        self.h = self.H
        t1 = time.time()
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')

        xforms = jt.array(self.transforms_gpu[img_ids]).unsqueeze(0)
        xforms = np.array(xforms)
        camera_dirs = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
            axis=-1)

        directions = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in xforms]

        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(directions, xforms)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(m):
            return [
                m * np.ones_like(origins[i][..., :1])
                for i in range(len(origins))
            ]

        lossmults = broadcast_scalar_attribute(1).copy()
        nears = broadcast_scalar_attribute(self.near).copy()
        fars = broadcast_scalar_attribute(self.far).copy()

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        origins = jt.array(origins)
        directions = jt.array(directions)
        viewdirs = jt.array(viewdirs)
        radii = jt.array(radii)
        lossmults = jt.array(lossmults)
        nears = jt.array(nears)
        fars = jt.array(fars)
        rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmults,
            near=nears,
            far=fars)
        del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs
        t2 = time.time()
        print("生成光线时间", t2 - t1)
        return rays


# @DATASETS.register_module()
# class Multicam:
#     def __init__(self, root_dir, batch_size, mode='train', H=0, W=0, near=0., far=1., img_alpha=True, have_img=True,
#                  preload_shuffle=True):
#         self.meta = None
#         self.white_bkgd = True
#         self.resolution = None
#         self.shuffle_index = None
#         self.rays = None
#         self.img_ids = None
#         self.data_dir = root_dir
#         self.batch_size = batch_size
#         self.preload_shuffle = preload_shuffle
#         self.H = H
#         self.W = W
#         self.transforms_gpu = []  # 读取transform_matrix
#         self.metadata = []
#         self.image_data = None
#         self.focal = None
#         self.n_images = 0
#         self.img_alpha = img_alpha  # img RGBA or RGB
#         assert mode == "train" or mode == "val" or mode == "test"
#         self.mode = mode
#         self.have_img = have_img
#         self.idx_now = 0
#         self.near = near
#         self.far = far
#         self.load_data()  # 初始化加载数据
#         jt.gc()
#
#     def __next__(self):
#         if self.idx_now + self.batch_size >= self.rays.origins.shape[0]:  # bs * iter 大于 n_images * h * w
#             rand_idx = jt.randperm(self.rays.origins.shape[0])
#             self.img_ids = self.img_ids[rand_idx]
#             self.rays = namedtuple_map(lambda r: r[rand_idx], self.rays)
#             self.image_data = self.image_data[rand_idx]
#             self.idx_now = 0
#         img_ids = self.img_ids[self.idx_now:self.idx_now + self.batch_size, 0].int()
#         rays = namedtuple_map(lambda r: jt.array(r[self.idx_now:self.idx_now + self.batch_size]), self.rays)
#         rgb_target = self.image_data[self.idx_now:self.idx_now + self.batch_size]
#         self.idx_now += self.batch_size
#         return img_ids, rays, rgb_target
#
#     def _flatten(self, x):
#         # 对gen_rays的rays做展平处理，因为外面包的是list
#         # Always flatten out the height x width dimensions
#         x = [y.reshape([-1, y.shape[-1]]) for y in x]
#         x = np.concatenate(x, axis=0)
#         # np.reshape(np.array(x), x[0].shape[-1])
#         return x
#
#     def _load_renderings(self):
#         print(f"load {self.mode} data")
#         with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as fp:
#             self.meta = json.load(fp)[self.split]
#         self.meta = {k: np.array(self.meta[k]) for k in self.meta}
#         # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
#         images = []
#         for relative_path in self.meta['file_path']:
#             image_path = os.path.join(self.data_dir, relative_path)
#             with open(image_path, 'rb') as image_file:
#                 image = np.array(Image.open(image_file), dtype=np.float32) / 255.
#             if self.white_bkgd:
#                 image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
#             images.append(image[..., :3])
#         self.images = images
#         del images
#         self.n_images = len(self.images)
#
#
# def _generate_rays(self):
#     """Generating rays for all images."""
#     pix2cam = self.meta['pix2cam'].astype(np.float32)
#     cam2world = self.meta['cam2world'].astype(np.float32)
#     width = self.meta['width'].astype(np.float32)
#     height = self.meta['height'].astype(np.float32)
#
#     def res2grid(w, h):
#         return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
#             np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
#             np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
#             indexing='xy')
#
#     xy = [res2grid(w, h) for w, h in zip(width, height)]
#     pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
#     camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
#     directions = [(v @ c2w[:3, :3].T).copy() for v, c2w in zip(camera_dirs, cam2world)]
#     origins = [
#         np.broadcast_to(c2w[:3, -1], v.shape).copy()
#         for v, c2w in zip(directions, cam2world)
#     ]
#     viewdirs = [
#         v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
#     ]
#
#     def broadcast_scalar_attribute(x):
#         return [
#             np.broadcast_to(x[i], origins[i][..., :1].shape).astype(np.float32)
#             for i in range(len(self.images))
#         ]
#
#     lossmult = broadcast_scalar_attribute(self.meta['lossmult']).copy()
#     near = broadcast_scalar_attribute(self.meta['near']).copy()
#     far = broadcast_scalar_attribute(self.meta['far']).copy()
#
#     # Distance from each unit-norm direction vector to its x-axis neighbor.
#     dx = [
#         np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
#     ]
#     dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
#     # Cut the distance in half, and then round it out so that it's
#     # halfway between inscribed by / circumscribed about the pixel.
#     radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
#
#     self.rays = Rays(
#         origins=origins,
#         directions=directions,
#         viewdirs=viewdirs,
#         radii=radii,
#         lossmult=lossmult,
#         near=near,
#         far=far)
#     del origins, directions, viewdirs, radii, lossmult, near, far, xy, pixel_dirs, camera_dirs
#
#     # TODO(bydeng): Swap this function with a more flexible camera model.
#     def _generate_rays_jt(self):
#         t1 = time.time()
#         """Generating rays for all images."""
#         x, y = jt.array(np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
#             np.arange(self.W, dtype=np.float32),  # X-Axis (columns)
#             np.arange(self.H, dtype=np.float32),  # Y-Axis (rows)
#             indexing='xy'))
#
#         camera_dirs = jt.stack(
#             [(x - self.W * 0.5 + 0.5) / self.focal_lengths[0][0],
#              -(y - self.H * 0.5 + 0.5) / self.focal_lengths[0][1], -jt.ones_like(x)],
#             -1)
#         directions = ((camera_dirs[None, ..., None, :] *
#                        self.transforms_gpu[:, None, None, :3, :3]).sum(-1))
#
#         origins = self.transforms_gpu[:, None, None, :3, -1].broadcast(
#             directions.shape)
#
#         viewdirs = directions / jt.norm(directions, dim=-1, keepdim=True)
#
#         # Distance from each unit-norm direction vector to its x-axis neighbor.
#         dx = jt.sqrt(
#             jt.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
#         dx = jt.concat([dx, dx[:, -2:-1, :]], 1)
#         # Cut the distance in half, and then round it out so that it's
#         # halfway between inscribed by / circumscribed about the pixel.
#         radii = dx[..., None] * 2 / jt.sqrt(12)  # 这个是像素中心点，映射成像素大小
#
#         ones = jt.ones_like(origins[..., :1])  # 这里是为了匹配rays的长度。都变成N_images * H * W
#         self.rays = Rays(
#             origins=origins.numpy(),
#             directions=directions.numpy(),
#             viewdirs=viewdirs.numpy(),
#             radii=radii.numpy(),
#             lossmult=ones.numpy(),
#             near=(ones * self.near).numpy(),
#             far=(ones * self.far).numpy())
#         t2 = time.time()
#         print(t2 - t1)
#
#     def _generate_rays(self):
#         """Generating rays for all images."""
#         self.w = self.W
#         self.h = self.H
#         t1 = time.time()
#         x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
#             np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
#             np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
#             indexing='xy')
#         camera_dirs = np.stack(
#             [(x - self.w * 0.5 + 0.5) / self.focal,
#              -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
#             axis=-1)
#
#         directions = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in self.transforms_gpu]
#
#         origins = [
#             np.broadcast_to(c2w[:3, -1], v.shape).copy()
#             for v, c2w in zip(directions, self.transforms_gpu)
#         ]
#         viewdirs = [
#             v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
#         ]
#
#         def broadcast_scalar_attribute(m):
#             return [
#                 m * np.ones_like(origins[i][..., :1])
#                 for i in range(self.n_images)
#             ]
#
#         lossmults = broadcast_scalar_attribute(1).copy()
#         nears = broadcast_scalar_attribute(self.near).copy()
#         fars = broadcast_scalar_attribute(self.far).copy()
#
#         # Distance from each unit-norm direction vector to its x-axis neighbor.
#         dx = [
#             np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
#         ]
#         dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
#         # Cut the distance in half, and then round it out so that it's
#         # halfway between inscribed by / circumscribed about the pixel.
#
#         radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
#
#         self.rays = Rays(
#             origins=origins,
#             directions=directions,
#             viewdirs=viewdirs,
#             radii=radii,
#             lossmult=lossmults,
#             near=nears,
#             far=fars)
#         del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs
#         t2 = time.time()
#         print("生成光线时间", t2 - t1)
#
#     def generate_rays_total_test(self, img_ids, H, W):
#         """Generating rays for all images.
#         这里创建的ray里面的各项，都是list
#         """
#         self.w = self.W
#         self.h = self.H
#         t1 = time.time()
#         x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
#             np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
#             np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
#             indexing='xy')
#
#         xforms = jt.array(self.transforms_gpu[img_ids]).unsqueeze(0)
#         xforms = np.array(xforms)
#         camera_dirs = np.stack(
#             [(x - self.w * 0.5 + 0.5) / self.focal,
#              -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
#             axis=-1)
#
#         directions = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in xforms]
#
#         origins = [
#             np.broadcast_to(c2w[:3, -1], v.shape).copy()
#             for v, c2w in zip(directions, xforms)
#         ]
#         viewdirs = [
#             v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
#         ]
#
#         def broadcast_scalar_attribute(m):
#             return [
#                 m * np.ones_like(origins[i][..., :1])
#                 for i in range(len(origins))
#             ]
#
#         lossmults = broadcast_scalar_attribute(1).copy()
#         nears = broadcast_scalar_attribute(self.near).copy()
#         fars = broadcast_scalar_attribute(self.far).copy()
#
#         # Distance from each unit-norm direction vector to its x-axis neighbor.
#         dx = [
#             np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
#         ]
#         dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
#         # Cut the distance in half, and then round it out so that it's
#         # halfway between inscribed by / circumscribed about the pixel.
#
#         radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
#
#         origins = jt.array(origins)
#         directions = jt.array(directions)
#         viewdirs = jt.array(viewdirs)
#         radii = jt.array(radii)
#         lossmults = jt.array(lossmults)
#         nears = jt.array(nears)
#         fars = jt.array(fars)
#         rays = Rays(
#             origins=origins,
#             directions=directions,
#             viewdirs=viewdirs,
#             radii=radii,
#             lossmult=lossmults,
#             near=nears,
#             far=fars)
#         del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs
#         t2 = time.time()
#         print("生成光线时间", t2 - t1)
#         return rays
