import logging
import jittor as jt
import cv2 as cv
import numpy as np
import os, json
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio
from utils import render_depth


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def calCenter(cameras:np.array):
    """Given cameras, calculate the center by interset the rays

    Args:
        cameras (np.array): [N,5,3]. The rays_o and 4 corners.
    """
    ray_o = cameras[:,0,:]  # [N,3]
    ray_d = cameras[:,1:,:].mean(axis=1) - ray_o  # [N,3]
    ray_d = ray_d / np.linalg.norm(ray_d, axis=1, keepdims=True)
    pts = []

    np.random.seed(1)  # set random seed, to avoid different every time
    for _ in range(100):
        id1, id2 = np.random.choice(np.arange(ray_o.shape[0]), 2, replace=False)
        ### calculate A * t = b
        A, b = np.zeros((2,2)), np.zeros(2,)
        A[0,0] = np.dot(ray_d[id1], ray_d[id1])
        A[0,1] = -1 * np.dot(ray_d[id1], ray_d[id2])
        A[1,0] = np.dot(ray_d[id1], ray_d[id2])
        A[1,1] = -1 * np.dot(ray_d[id2], ray_d[id2])
        b[0] = np.dot(ray_o[id2]-ray_o[id1], ray_d[id1])
        b[1] = np.dot(ray_o[id2]-ray_o[id1], ray_d[id2])
        try:
            t = np.linalg.solve(A, b)
            pts.append(ray_o[[id1,id2]] + ray_d[[id1,id2]] * t[:,np.newaxis])
        except:
            print("unable to solve ...")
            continue
    pts = np.concatenate(pts, axis=0)  # [20, 3]
    # print("center", pts.mean(axis=0))
    return pts.mean(axis=0)

def normalizeCamera(c2w, H, W, F, rad=3):
    ### construct camera: [N,5,3]
    isTensor = False
    if jt.is_var(c2w):
        c2w_np = c2w.cpu().numpy()
        isTensor = True
    else:
        c2w_np = c2w
    
    rays_o = c2w_np[:,:3,3]  # [N,3]

    left_top = [-1*W/2, H/2]
    right_top = [W/2, H/2]
    left_bottom = [-1*W/2, -1*H/2]
    right_bottom = [W/2, -1*H/2]
    corners = np.stack([left_bottom,right_bottom,right_top,left_top], axis=0)  # [NumPt,2]
    corners /= F
    corners = np.concatenate([corners, -1*np.ones_like(corners[:,:1])], axis=-1)  # [NumPt,3]
    # change the z direction to negative
    corners = np.broadcast_to(corners, [c2w_np.shape[0]]+list(corners.shape))  # [N,NumPt,3]
    rays_d = np.matmul(c2w_np[:,:3,:3], corners.transpose(0,2,1)).transpose(0,2,1)  # [N,NumPt,3]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)  # [N,NumPt,3]
    cameras = np.concatenate([rays_o[:,np.newaxis,:], rays_o[:,np.newaxis,:]+rays_d], axis=1)
    center = calCenter(cameras)  # the center of cameras
    cameras -= center  # move to origin
    radius = np.linalg.norm(cameras[:,0,:], axis=-1).mean()
    scale = rad / radius  # change different scale of inside ball 

    c2w_np[:,:3,3] -= center
    c2w_np[:,:3,3] *= scale

    if isTensor:
        c2w_np = jt.array(c2w_np)

    return c2w_np


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        jt.flags.use_cuda = 1
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        dataset_type = conf.get_string('type')

        USE_CUSTOM_DATASET = False
        if dataset_type == 'nerf_synthetic':
            print("Use the nerf synthetic dataset")
            splits = ['train', 'val', 'test']
            metas = {}
            for s in splits:
                with open(os.path.join(self.data_dir, 'transforms_{}.json'.format(s)), 'r') as fp:
                    metas[s] = json.load(fp)

            train_meta = metas['train']
            img_files = []
            all_imgs = []
            all_poses = []
            all_val_poses = []
            for frame in train_meta['frames']:
                fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
                img_files.append(fname)
                all_imgs.append(imageio.imread(fname))
                all_poses.append(np.array(frame['transform_matrix']))
            all_imgs = (np.array(all_imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            all_poses = np.array(all_poses).astype(np.float32)
            all_val_poses = all_poses[-5:]

            H, W = all_imgs[0].shape[:2]
            camera_angle_x = float(train_meta['camera_angle_x'])
            focal = .5 * W / np.tan(.5 * camera_angle_x)
            _intrinsics = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        elif dataset_type == 'custom':
            print("Use the custom dataset")
            USE_CUSTOM_DATASET = True
            NEED_INVERT = False
            pose_dir = os.path.join(self.data_dir, "pose")
            if not os.path.exists(pose_dir):
                pose_dir = os.path.join(self.data_dir, "extrinsic")
                NEED_INVERT = True
            img_dir = os.path.join(self.data_dir, "rgb")

            pose_files = sorted([os.path.join(pose_dir, x) for idx,x in enumerate(os.listdir(pose_dir))])
            img_files = sorted([os.path.join(img_dir, x) for idx,x in enumerate(os.listdir(img_dir))])
            val_pose_files = pose_files[-5:]

            all_poses = [np.loadtxt(x) for x in sorted(pose_files)] # list of 4x4 array
            all_val_poses = [np.loadtxt(x) for x in sorted(val_pose_files)] # list of 4x4 array

            if NEED_INVERT:
                all_poses = [np.linalg.inv(x) for x in all_poses]
                all_val_poses = [np.linalg.inv(x) for x in all_val_poses]

            all_poses = np.stack(all_poses, axis=0).astype(np.float32)
            all_val_poses = np.stack(all_val_poses, axis=0).astype(np.float32)

            all_poses[:,:,1:3] = -all_poses[:,:,1:3] # I don't know why ...
            all_val_poses[:,:,1:3] = -all_val_poses[:,:,1:3] # I don't know why ...

            all_imgs = [imageio.imread(x) for x in sorted(img_files)] # list of images
            all_imgs = np.stack(all_imgs, axis=0).astype(np.float32) / 255. # keep all 3 channels (RGBA):3

            intrinsic_path = os.path.join(self.data_dir, "intrinsics.txt") # 
            with open(intrinsic_path) as f:
                lines = f.readlines()
            focal = np.fromstring(lines[0], sep=' ', dtype=np.float32)[0]
            H, W =  np.fromstring(lines[-1], sep=' ', dtype=np.int)
            _intrinsics = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

            print("normalize the camera")
            all_poses = normalizeCamera(all_poses, H, W, focal)

        else:
            raise NotImplementedError

        ### load data
        self.n_images = len(img_files)
        self.images_np = all_imgs[..., [2,1,0]]
        self.masks_np = all_imgs[..., 3:]
        if USE_CUSTOM_DATASET:
            self.masks_np = np.ones_like(all_imgs[..., 0:1])
        self.images_lis = img_files

        self.intrinsics_all = [jt.array(_intrinsics).float()] * self.n_images
        self.pose_all = [jt.array(pose).float() for pose in all_poses]
        self.val_pose_all = [jt.array(pose).float() for pose in all_val_poses]

        self.images = jt.array(self.images_np.astype(np.float32))  # [n_images, H, W, 3]
        self.masks  = jt.array(self.masks_np.astype(np.float32))   # [n_images, H, W, 3]
        self.intrinsics_all = jt.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = jt.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = jt.stack(self.pose_all)  # [n_images, 4, 4]
        self.val_pose_all = jt.stack(self.val_pose_all)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.3, -1.3, -1.3, 1.0])
        object_bbox_max = np.array([ 1.3,  1.3,  1.3, 1.0])
        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]

        print('Load data: End')


    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        img_idx = int(img_idx)
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        K = self.intrinsics_all[img_idx]
        rays_v = jt.stack([(pixels_x-K[0][2])/K[0][0], -(pixels_y-K[1][2])/K[1][1], -jt.ones_like(pixels_x)], -1)  # [W,H,3]
        rays_v = jt.matmul(self.pose_all[img_idx, None, None, :3, :3].expand(self.W//l, self.H//l, 3,3), rays_v[:, :, :, None]).squeeze(-1)  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        img_idx = int(img_idx)
        pixels_x = jt.randint(low=0, high=self.W, shape=(batch_size,))
        pixels_y = jt.randint(low=0, high=self.H, shape=(batch_size,))
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3 
        K = self.intrinsics_all[img_idx]
        rays_v = jt.stack([(pixels_x-K[0][2])/K[0][0], -(pixels_y-K[1][2])/K[1][1], -jt.ones_like(pixels_x)], -1)  # batch_size, 3
        rays_v = jt.matmul(self.pose_all[img_idx, None, :3, :3].expand(batch_size,3,3), rays_v[:, :, None]).squeeze(-1)  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return jt.concat([rays_o, rays_v, color, mask[:, :1]], dim=-1)    # batch_size, 10


    def near_far_from_sphere(self, rays_o, rays_d):
        a = jt.sum(rays_d**2, dim=-1, keepdims=True)
        b = 2.0 * jt.sum(rays_o * rays_d, dim=-1, keepdims=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

### ----------------------------------------------------------

    def gen_circle_poses(self):
        trans_t = lambda t : jt.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,t],
            [0,0,0,1]]).float()

        rot_phi = lambda phi : jt.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1]]).float()

        rot_theta = lambda th : jt.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1]]).float()

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
            return c2w

        Nframes = 40

        if "hbychair" in self.data_dir:
            print("set render raids to 3.5 ! Add aditional rotation.")
            render_poses = []
            for elev in [-30]:
                render_pose = jt.stack([pose_spherical(angle, elev, 3.5) for angle in np.linspace(-180,180,Nframes+1)[:-1]], 0)
                rotation_mat = jt.array([[1,0,0,0],[0,-0.35,0.94,0],[0,-0.94,-0.35,-0.15],[0,0,0,1]]).float()
                rotation_mat = jt.linalg.inv(rotation_mat)
                render_poses += [rotation_mat @ x for x in render_pose]
            render_poses = jt.stack(render_poses)
        else:
            print("set render raids to 3.5 !")
            for elev in [-30]:
                render_poses = jt.stack([pose_spherical(angle, elev, 3.5) for angle in np.linspace(-180,180,Nframes+1)[:-1]], 0)

        return render_poses

    def gen_validation_pose(self):
        return self.val_pose_all[::5]

    def gen_rays_at_pose(self, render_pose, resolution_level=1):
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)  # [W,H]
        K = self.intrinsics_all[0]
        rays_v = jt.stack([(pixels_x-K[0][2])/K[0][0], -(pixels_y-K[1][2])/K[1][1], -jt.ones_like(pixels_x)], -1)  # [W,H,3]
        w, h = rays_v.shape[:2]
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = jt.matmul(render_pose[None, None, :3, :3].expand(w,h,3,3), rays_v[:, :, :, None]).squeeze(-1)  # W, H, 3
        rays_o = render_pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)



    def gen_rays_at_pose_with_depth(self, render_pose, mesh, resolution_level=1):
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)  # [W,H]
        K = self.intrinsics_all[0]
        
        rays_v = jt.stack([(pixels_x-K[0][2])/K[0][0], -(pixels_y-K[1][2])/K[1][1], -jt.ones_like(pixels_x)], -1)  # [W,H,3]
        w, h = rays_v.shape[:2]
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = jt.matmul(render_pose[None, None, :3, :3].expand(w,h,3,3), rays_v[:, :, :, None]).squeeze(-1)  # W, H, 3
        rays_o = render_pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3

        # dep_norm_render = DepthNormalRenderer(render_pose.unsqueeze(0), IMGSIZE=self.H//l, \
        #             FOCAL=K.cpu()[0][0]/l, aspect_ratio=float(self.W)/float(self.H))
        # dep_im, norm_im = dep_norm_render(mesh)  # [bs,H,W,1], [bs,H,W,3]
        dep_im = render_depth(render_pose.numpy(), mesh, IMGSIZE=self.H//l, FOCAL=K.numpy()[0][0]/l, aspect_ratio=float(self.W)/float(self.H))
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), dep_im

