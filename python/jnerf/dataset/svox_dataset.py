# Standard NeRF Blender dataset loader

from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np
import jittor as jt
import jittor.nn as nn
from typing import Union, Optional, List
from jnerf.utils.svox2_utils import select_or_shuffle_rays,RaysDataset,Intrin
from jnerf.utils.registry import DATASETS

@DATASETS.register_module()
class DatasetBase:
    split: str
    permutation: bool
    epoch_size: Optional[int]
    n_images: int
    h_full: int
    w_full: int
    intrins_full: Intrin
    c2w: jt.Var  # C2W OpenCV poses
    gt: Union[jt.Var, List[jt.Var]]   # RGB images
    # device : Union[str, torch.device]

    def __init__(self):
        self.ndc_coeffs = (-1, -1)
        self.use_sphere_bound = False
        self.should_use_background = True # a hint
        self.use_sphere_bound = True
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.permutation = False

    def shuffle_rays(self):
        """
        Shuffle all rays
        """
        if self.split == "train":
            del self.rays
            self.rays = select_or_shuffle_rays(self.rays_init, self.permutation,
                                               self.epoch_size)

    def gen_rays(self, factor=1):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins = self.intrins_full.scale(1.0 / true_factor)
        yy, xx = jt.meshgrid(jt.arange(self.h,dtype=jt.float32)+0.5,
                             jt.arange(self.w,dtype=jt.float32)+0.5)
        xx = (xx - self.intrins.cx) / self.intrins.fx
        yy = (yy - self.intrins.cy) / self.intrins.fy
        zz = jt.ones_like(xx)
        dirs = jt.stack((xx, yy, zz), dim=-1) # OpenCV convention
        dirs /= jt.norm(dirs,dim=-1,keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
        del xx, yy, zz
        dirs = (self.c2w[:, None, :3, :3].repeat(1,dirs.shape[1],1,1) @ dirs.repeat(self.c2w.shape[0],1,1,1))[..., 0]
        if factor != 1:
            gt = nn.interpolate(
                self.gt.permute([0,3,1,2]),size=(self.h,self.w),mode="area"
            ).permute([0,2,3,1])
            gt = gt.reshape(self.n_images, -1, 3)
        else:
            gt = self.gt.reshape(self.n_images, -1, 3)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1)
        if self.split == "train":
            origins = origins.view(-1, 3)
            dirs = dirs.view(-1, 3)
            gt = gt.reshape(-1, 3)

        self.rays_init = RaysDataset(origins=origins, dirs=dirs, gt=gt)
        self.rays = self.rays_init

    def get_image_size(self, i : int):
        # H, W
        if hasattr(self, 'image_size'):
            return tuple(self.image_size[i])
        else:
            return self.h, self.w

@DATASETS.register_module()
class SvoxNeRFDataset(DatasetBase):
    """
    NeRF dataset loader
    """

    focal: float
    c2w: jt.Var # (n_images, 4, 4)
    gt: jt.Var  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[RaysDataset]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        scene_scale: Optional[float] = None,
        factor: int = 1,
        scale : Optional[float] = None,
        # permutation: bool = True,
        white_bkgd: bool = True,
        n_images = None,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 2/3
        if scale is None:
            scale = 1.0
        self.epoch_size = epoch_size
        all_c2w = []
        all_gt = []

        split_name = split if split != "test_train" else "train"
        data_path = path.join(root, split_name)
        data_json = path.join(root, "transforms_" + split_name + ".json")

        print("LOAD DATA", data_path)

        j = json.load(open(data_json, "r"))

        # OpenGL -> OpenCV
        cam_trans = jt.diag(jt.array([1, -1, -1, 1],dtype=jt.float32))

        for frame in tqdm(j["frames"]):
            fpath = path.join(data_path, path.basename(frame["file_path"]) + ".png")
            c2w = jt.array(frame["transform_matrix"],dtype=jt.float32)
            c2w = c2w @ cam_trans  # To OpenCV

            im_gt = imageio.imread(fpath)
            if scale < 1.0:
                full_size = list(im_gt.shape[:2])
                rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

            all_c2w.append(c2w)
            all_gt.append(jt.array(np.array(im_gt)))
        focal = float(
            0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
        )
        self.c2w = jt.stack(all_c2w)
        self.c2w[:, :3, 3] *= scene_scale

        self.gt = jt.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        # Choose a subset of training images
        if n_images is not None:
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images,...]
            self.c2w = self.c2w[0:n_images,...]

        self.intrins_full : Intrin = Intrin(focal, focal,
                                            self.w_full * 0.5,
                                            self.h_full * 0.5)



        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

        self.should_use_background = False  # Give warning

