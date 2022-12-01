
import random
import jittor as jt
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
from .dataset_util import *

@DATASETS.register_module()
class NerfDataset():

    def __init__(self,root_dir, batch_size, mode='train', H=0, W=0, correct_pose=[1,-1,-1], aabb_scale=None, scale=None, offset=None, img_alpha=True,to_jt=True, have_img=True, preload_shuffle=True):
        self.root_dir=root_dir
        self.batch_size=batch_size
        self.preload_shuffle=preload_shuffle
        self.H=H
        self.W=W
        self.correct_pose = correct_pose
        self.aabb_scale = aabb_scale
        if scale is None:
            self.scale = NERF_SCALE
        else:
            self.scale = scale
        if offset is None:
            self.offset=[0.5,0.5,0.5]
        else:
            self.offset=offset
        self.resolution=[0,0]# W*H
        self.transforms_gpu=[]
        self.metadata=[]
        self.image_data=[]
        self.focal_lengths=[]
        self.n_images=0
        self.img_alpha=img_alpha# img RGBA or RGB
        self.to_jt=to_jt
        self.have_img=have_img
        self.compacted_img_data=[]# img_id ,rgba,ray_d,ray_o
        assert mode=="train" or mode=="val" or mode=="test"
        self.mode=mode
        self.idx_now=0
        self.load_data()
        jt.gc()
        self.image_data = self.image_data.reshape(
            self.n_images, -1, 4).detach()
        # breakpoint()

    def __next__(self):
        if self.idx_now+self.batch_size >= self.shuffle_index.shape[0]:
            del self.shuffle_index
            self.shuffle_index=jt.randperm(self.n_images*self.H*self.W).detach()
            jt.gc()
            self.idx_now = 0      
        img_index=self.shuffle_index[self.idx_now:self.idx_now+self.batch_size]
        img_ids,rays_o,rays_d,rgb_target=self.generate_random_data(img_index,self.batch_size)
        self.idx_now+=self.batch_size
        return img_ids, rays_o, rays_d, rgb_target
        
    def load_data(self,root_dir=None):
        print(f"load {self.mode} data")
        if root_dir is None:
            root_dir=self.root_dir
        ##get json file
        json_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1] == ".json":
                    if self.mode in os.path.splitext(file)[0] or (self.mode=="train" and "val" in os.path.splitext(file)[0]):
                        json_paths.append(os.path.join(root, file))
        json_data=None
        ## get frames
        for json_path in json_paths:

            with open(json_path,'r')as f:
                data=json.load(f)
            if json_data is None:
                json_data=data
            else:
                json_data['frames']+=data['frames']

        assert json_data is not None, f"dataset is not found at {root_dir}"
        ## init set  scale & offset
        if 'h' in json_data:
            self.H=int(json_data['h'])
        if 'w' in json_data:
            self.W=int(json_data['w'])

        frames=json_data['frames']
        if self.mode=="val":
            frames=frames[::10]

        for frame in tqdm(frames):
            if self.have_img:
                img_path=os.path.join(self.root_dir,frame['file_path'])
                if not os.path.exists(img_path):
                    img_path=img_path+".png"
                    if not os.path.exists(img_path):
                        continue
                img = read_image(img_path)
                if self.H==0 or self.W==0:
                    self.H=int(img.shape[0])
                    self.W=int(img.shape[1])
                self.image_data.append(img)
            else:
                self.image_data.append(np.zeros((self.H,self.W,3)))
            self.n_images+=1
            matrix=np.array(frame['transform_matrix'],np.float32)[:-1, :]
            self.transforms_gpu.append(
                            self.matrix_nerf2ngp(matrix, self.scale, self.offset))
                           
        self.resolution=[self.W,self.H]
        self.resolution_gpu=jt.array(self.resolution)
        metadata=np.empty([11],np.float32)
        metadata[0]=json_data.get('k1',0)
        metadata[1]=json_data.get('k2',0)
        metadata[2]= json_data.get('p1',0)
        metadata[3]=json_data.get('p2',0)
        metadata[4]=json_data.get('cx',self.W/2)/self.W
        metadata[5]=json_data.get('cy',self.H/2)/self.H
        def read_focal_length(resolution: int, axis: str):
            if 'fl_'+axis in json_data:
                return json_data['fl_'+axis]
            elif 'camera_angle_'+axis in json_data:
                return fov_to_focal_length(resolution, json_data['camera_angle_'+axis]*180/pi)
            else:
                return 0
        x_fl = read_focal_length(self.resolution[0], 'x')
        y_fl = read_focal_length(self.resolution[1], 'y')
        focal_length = []
        if x_fl != 0:
            focal_length = [x_fl, x_fl]
            if y_fl != 0:
                focal_length[1] = y_fl
        elif y_fl != 0:
            focal_length = [y_fl, y_fl]
        else:
            raise RuntimeError("Couldn't read fov.")
        self.focal_lengths.append(focal_length)
        metadata[6]=focal_length[0]
        metadata[7]=focal_length[1]

        light_dir=np.array([0,0,0])
        metadata[8:]=light_dir
        self.metadata =np.expand_dims(metadata,0).repeat(self.n_images,axis=0)
        if self.aabb_scale is None:
            self.aabb_scale=json_data.get('aabb_scale',1)
        aabb_range=(0.5,0.5)
        self.aabb_range=(aabb_range[0]-self.aabb_scale/2,aabb_range[1]+self.aabb_scale/2)
        self.H=int(self.H)
        self.W=int(self.W)

        self.image_data=jt.array(self.image_data)
        self.transforms_gpu=jt.array(self.transforms_gpu)
        self.focal_lengths=jt.array(self.focal_lengths).repeat(self.n_images,1)
        ## transpose to adapt Eigen::Matrix memory
        self.transforms_gpu=self.transforms_gpu.transpose(0,2,1)
        self.metadata=jt.array(self.metadata)
        if self.img_alpha and self.image_data.shape[-1]==3:
            self.image_data=jt.concat([self.image_data,jt.ones(self.image_data.shape[:-1]+(1,))],-1).stop_grad()
        self.shuffle_index=jt.randperm(self.H*self.W*self.n_images).detach()
        jt.gc()
    
    def generate_random_data(self,index,bs):
        img_id=index//(self.H*self.W)
        img_offset=index%(self.H*self.W)
        focal_length =self.focal_lengths[img_id]
        xforms = self.transforms_gpu[img_id]
        principal_point = self.metadata[:, 4:6][img_id]
        xforms=xforms.permute(0,2,1)
        rays_o = xforms[...,  3]
        res = self.resolution_gpu
        x=((img_offset%self.W)+0.5)/self.W
        y=((img_offset//self.W)+0.5)/self.H
        xy=jt.stack([x,y],dim=-1)
        rays_d = jt.concat([(xy-principal_point)* res/focal_length, jt.ones([bs, 1])], dim=-1)
        rays_d = jt.normalize(xforms[ ...,  :3].matmul(rays_d.unsqueeze(2)))
        rays_d=rays_d.squeeze(-1)
        rgb_tar=self.image_data.reshape(-1,4)[index]
        return img_id,rays_o,rays_d,rgb_tar

    def generate_rays_total(self, img_id,H,W):
        H=int(H)
        W=int(W)
        img_size=H*W
        focal_length =self.focal_lengths[img_id]
        xforms = self.transforms_gpu[img_id]
        principal_point = self.metadata[:, 4:6][img_id]
        xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
                      W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        # assert H==W
        # xy += (jt.rand_like(xy)-0.5)/H
        xforms=xforms.permute(1,0)
        rays_o = xforms[:,  3]
        res = jt.array(self.resolution)
        rays_d = jt.concat([(xy-principal_point)* res/focal_length, jt.ones([H*W, 1])], dim=-1)
        rays_d = jt.normalize(xforms[ :,  :3].matmul(rays_d.unsqueeze(2)))
        rays_d=rays_d.squeeze(-1)
        return rays_o, rays_d

    def generate_rays_total_test(self, img_ids, H, W):
        # select focal,trans,p_point
        focal_length = jt.gather(
            self.focal_lengths, 0, img_ids)
        xforms = jt.gather(self.transforms_gpu, 0, img_ids)
        principal_point = jt.gather(
            self.metadata[:, 4:6], 0, img_ids)
        # rand generate uv 0~1
        xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
                      W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        # assert H==W
        # xy += (jt.rand_like(xy)-0.5)/H
        xy_int = jt.stack(jt.meshgrid(jt.linspace(
            0, H-1, H), jt.linspace(0, W-1, W)), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        xforms=xforms.fuse_transpose([0,2,1])
        rays_o = jt.gather(xforms, 0, img_ids)[:, :, 3]
        res = jt.array(self.resolution)
        rays_d = jt.concat([(xy-jt.gather(principal_point, 0, img_ids))
                           * res/focal_length, jt.ones([H*W, 1])], dim=-1)
        rays_d = jt.normalize(jt.gather(xforms, 0, img_ids)[
                              :, :, :3].matmul(rays_d.unsqueeze(2)))
        # resolution W H
        # img H W
        rays_pix = ((xy_int[:, 1]) * H+(xy_int[:, 0])).int()
        # rays origin /dir   rays hit point offset
        return rays_o, rays_d, rays_pix
    
    def generate_rays_with_pose(self, pose, H, W):
        nray = H*W
        pose = self.matrix_nerf2ngp(pose, self.scale, self.offset)
        focal_length = self.focal_lengths[:1].expand(nray, -1)
        xforms = pose.unsqueeze(0).expand(nray, -1, -1)
        principal_point = self.metadata[:1, 4:6].expand(nray, -1)
        xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
                      W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        xy_int = jt.stack(jt.meshgrid(jt.linspace(
            0, H-1, H), jt.linspace(0, W-1, W)), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        rays_o = xforms[:, :, 3]
        res = jt.array(self.resolution)
        rays_d = jt.concat([
            (xy-principal_point) * res/focal_length, 
            jt.ones([H*W, 1])
        ], dim=-1)
        rays_d = jt.normalize(xforms[:, :, :3].matmul(rays_d.unsqueeze(2)))
        return rays_o, rays_d

    def matrix_nerf2ngp(self, matrix, scale, offset):
        matrix[:, 0] *= self.correct_pose[0]
        matrix[:, 1] *= self.correct_pose[1]
        matrix[:, 2] *= self.correct_pose[2]
        matrix[:, 3] = matrix[:, 3] * scale + offset
        # cycle
        matrix=matrix[[1,2,0]]
        return matrix

    def matrix_ngp2nerf(self, matrix, scale, offset):
        matrix=matrix[[2,0,1]]
        matrix[:, 0] *= self.correct_pose[0]
        matrix[:, 1] *= self.correct_pose[1]
        matrix[:, 2] *= self.correct_pose[2]
        matrix[:, 3] = (matrix[:, 3] - offset) / scale
        return matrix