# This file is modified from official mipnerf
"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
import time
from os import path
import cv2
import numpy as np
from PIL import Image
import collections
from jnerf.utils.registry import DATASETS
import jittor as jt

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


class BaseDataset:
    """BaseDataset Base Class."""

    def __init__(self, data_dir, split, batch_size=4096, batch_type="all_images", white_bkgd=True):
        self.near = 2
        self.far = 6
        self.split = split
        self.data_dir = data_dir
        self.white_bkgd = white_bkgd
        self.batch_type = batch_type
        self.images = None  # 图片数据
        self.rays = None  # 光线数据
        self.it = -1
        self.n_examples = 1  # 多少张图片
        self.factor = 0
        self.idx_now = 0
        self.batch_size = batch_size
        jt.gc()

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            x = np.concatenate(x, axis=0)
        return jt.array(x)

    def _train_init(self):
        """Initialize training."""

        self._load_renderings()
        self._generate_rays()

        if self.split == 'train':
            if self.batch_type == 'all_images':  # 'The batch_type can only be all_images with flatten'
                # flatten the ray and image dimension together.
                self.images = self._flatten(self.images)  # 像素展平了, 还把list变成了array
                self.rays = namedtuple_map(self._flatten, self.rays)
            else:
                print("像素不展平，把list变成array，开始从每张图中抽取bs个像素训练")
                self.images = self._flatten(self.images)  # 100 h*w c
                self.rays = namedtuple_map(self._flatten, self.rays)
        else:
            assert self.batch_type == 'single_image', 'The batch_type can only be single_image without flatten'

    def _val_init(self):
        # 没有flatten,
        self._load_renderings()  # """Load images from disk.""" 这里有image
        self._generate_rays()  # 这里是光线

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_renderings(self):
        raise ValueError('Implement in different dataset.')

    def __next__(self):
        if self.batch_type == 'all_images':
            if self.idx_now + self.batch_size >= self.rays.origins.shape[0]:
                rand_idx = jt.randperm(self.rays.origins.shape[0])  # bs * iter 大于 n_images * h * w
                t3 = time.time()
                self.rays = namedtuple_map(lambda r: r[rand_idx], self.rays)
                self.images = self.images[rand_idx]
                self.idx_now = 0
                t4 = time.time()
                print("shuffle data need time: {}s".format((t4 - t3)))
            rays = namedtuple_map(lambda r: jt.array(r[self.idx_now:self.idx_now + self.batch_size]), self.rays)
            rgb_target = self.images[self.idx_now:self.idx_now + self.batch_size]
            # ray_indices = np.random.randint(0, self.rays[0].shape[0], (self.batch_size,))
            # rays = namedtuple_map(lambda r: jt.array(r[ray_indices]), self.rays)
            # rgb_target = self.images[ray_indices]
            self.idx_now += self.batch_size
        else:
            # (jax上是随机抽一张图）
            image_index = np.random.randint(0, self.n_examples)
            ray_indices = np.random.randint(0, self.rays[0][0].shape[0], (self.batch_size,))
            rgb_target = self.images[image_index][ray_indices]
            rays = namedtuple_map(lambda r: jt.array(r[image_index][ray_indices]), self.rays)
            # 依次从单图中抽取bs个像素
        return rays, rgb_target  # 返回rays 和 images


@DATASETS.register_module()
class Multicam(BaseDataset):
    """Multicam Dataset."""

    def __init__(self, data_dir, split, batch_size=4096, batch_type="all_images", white_bkgd=True):
        super(Multicam, self).__init__(data_dir, split, batch_size, batch_type, white_bkgd)
        t1 = time.time()
        if split == 'train':  # 如果all——images，都把list变成了array，如果sinal-image，还是list
            self._train_init()
            # 初始打乱一下
            rand_idx = jt.randperm(self.rays.origins.shape[0])  # bs * iter 大于 n_images * h * w
            t3 = time.time()
            self.rays = namedtuple_map(lambda r: r[rand_idx], self.rays)
            self.images = self.images[rand_idx]
            t4 = time.time()
            print("shuffle data need time: {}s".format((t4 - t3)))
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()
        t2 = time.time()
        print("dataloader {} datasets need {}s".format(split, (t2 - t1)))

    def _load_renderings(self):
        """Load images from disk."""
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as fp:
            self.meta = json.load(fp)[self.split]
        if self.split == 'train':
            self.meta = {k: np.array(self.meta[k]) for k in self.meta}
        else:
            self.meta = {k: np.array(self.meta[k][:20]) for k in self.meta}  # 只取前20进行测试和验证
        # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
        images = []
        for relative_path in self.meta['file_path']:
            image_path = os.path.join(self.data_dir, relative_path)
            with open(image_path, 'rb') as image_file:
                image = np.array(Image.open(image_file), dtype=np.float32) / 255.
            if self.white_bkgd:
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            images.append(image[..., :3])
        self.images = images
        del images
        self.n_examples = len(self.images)

    def _generate_rays(self):
        """Generating rays for all images."""
        pix2cam = self.meta['pix2cam'].astype(np.float32)
        cam2world = self.meta['cam2world'].astype(np.float32)
        width = self.meta['width'].astype(np.float32)
        height = self.meta['height'].astype(np.float32)

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                indexing='xy')

        xy = [res2grid(w, h) for w, h in zip(width, height)]
        pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
        camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
        directions = [(v @ c2w[:3, :3].T).copy() for v, c2w in zip(camera_dirs, cam2world)]
        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(directions, cam2world)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return [
                np.broadcast_to(x[i], origins[i][..., :1].shape).astype(np.float32)
                for i in range(len(self.images))
            ]

        lossmult = broadcast_scalar_attribute(self.meta['lossmult']).copy()
        near = broadcast_scalar_attribute(self.meta['near']).copy()
        far = broadcast_scalar_attribute(self.meta['far']).copy()

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
            lossmult=lossmult,
            near=near,
            far=far)
        del origins, directions, viewdirs, radii, lossmult, near, far, xy, pixel_dirs, camera_dirs


@DATASETS.register_module()
class Blenders(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, split, batch_size=4096, batch_type="all_images", white_bkgd=True, factor=0):
        super(Blenders, self).__init__(data_dir, split, batch_size, batch_type, white_bkgd)
        t1 = time.time()
        if split == 'train':
            self._train_init()
            # # 初始打乱一下
            rand_idx = jt.randperm(self.rays.origins.shape[0])  # bs * iter 大于 n_images * h * w
            t3 = time.time()
            self.rays = namedtuple_map(lambda r: r[rand_idx], self.rays)
            self.images = self.images[rand_idx]
            t4 = time.time()
            print("shuffle data need time: {}s".format((t4 - t3)))
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()
        t2 = time.time()
        print("dataloader {} datasets need {}s".format(split, (t2 - t1)))

    def _load_renderings(self):
        """Load images from disk."""
        with open(path.join(self.data_dir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
        images = []
        cams = []
        if self.split == 'train':
            pass
        else:
            meta['frames'] = meta['frames'][:20]
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
                elif self.factor > 0:
                    raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                                     'set.'.format(self.factor))
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            if self.white_bkgd:
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            images.append(image[..., :3])

        self.images = images
        del images
        self.h, self.w = self.images[0].shape[:-1]
        self.camtoworlds = cams
        del cams
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.n_examples = len(self.images)

    def _generate_rays(self):
        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
            axis=-1)

        directions = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in self.camtoworlds]

        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(directions, self.camtoworlds)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return [
                x * np.ones_like(origins[i][..., :1])
                for i in range(len(self.images))
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


class RealData360(BaseDataset):
    """RealData360 Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', factor=0):
        super(RealData360, self).__init__(data_dir, split, white_bkgd, batch_type, factor)
        if split == 'train':
            self._train_init()
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()

    def _load_renderings(self):
        """Load images from disk."""
        # Load images.
        imgdir_suffix = ''
        if self.factor > 0:
            imgdir_suffix = '_{}'.format(self.factor)
        else:
            factor = 1
        imgdir = path.join(self.data_dir, 'images' + imgdir_suffix)
        if not path.exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in sorted(os.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images = []
        for imgfile in imgfiles:
            with open(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                images.append(image)
        images = np.stack(images, axis=-1)

        # Load poses and bds.
        with open(path.join(self.data_dir, 'poses_bounds.npy'), 'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1]:
            raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
                images.shape[-1], poses.shape[-1]))

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / self.factor

        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Recenter poses.
        poses = self._recenter_poses(poses)
        poses = self._spherify_poses(poses)
        # Select the split.
        i_test = np.arange(images.shape[0])[::8]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test])
        if self.split == 'train':
            indices = i_train
        else:
            indices = i_test
        images = images[indices]
        poses = poses[indices]
        bds = bds[indices]
        self._read_camera()
        self.K[:2, :] /= self.factor
        self.K_inv = np.linalg.inv(self.K)
        self.K_inv[1:, :] *= -1
        self.bds = bds
        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        self.n_examples = images.shape[0]

    def _generate_rays(self):
        """Generating rays for all images."""

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                indexing='xy')

        xy = res2grid(self.w, self.h)
        pixel_dirs = np.stack([xy[0], xy[1], np.ones_like(xy[0])], axis=-1)
        camera_dirs = pixel_dirs @ self.K_inv.T
        directions = ((camera_dirs[None, ..., None, :] * self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                                  directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.sqrt(
            np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)
        ones = np.ones_like(origins[..., :1])
        near_fars = np.broadcast_to(self.bds[:, None, None, :], [*directions.shape[:-1], 2])
        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=near_fars[..., 0:1],
            far=near_fars[..., 1:2])
        del origins, directions, viewdirs, radii, near_fars, ones, xy, pixel_dirs, camera_dirs

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def _read_camera(self):
        import struct
        # modified from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py

        def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
            """Read and unpack the next bytes from a binary file.
            :param fid:
            :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
            :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
            :param endian_character: Any of {@, =, <, >, !}
            :return: Tuple of read and unpacked values.
            """
            data = fid.read(num_bytes)
            return struct.unpack(endian_character + format_char_sequence, data)

        with open(path.join(self.data_dir, 'sparse', '0', 'cameras.bin'), "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            num_params = 4
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            self.K = np.array([[params[0], 0, params[2]],
                               [0, params[1], params[3]],
                               [0, 0, 1]])

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _spherify_poses(self, poses):
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        return poses_reset
