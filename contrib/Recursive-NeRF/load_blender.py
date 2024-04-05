import os
import numpy as np
import imageio 
import json
import jittor as jt
import cv2


trans_t = lambda t : jt.array(np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).astype(np.float32))

rot_phi = lambda phi : jt.array(np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(np.float32))

rot_theta = lambda th : jt.array(np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(np.float32))


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, factor=1, do_intrinsic = False):
    if half_res and factor==1:
        factor=2
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    if do_intrinsic:
        a=np.array(meta['intrinsic_matrix'])
        # H, W, focal = 480,640,585
        # a = np.eye(4).astype(np.float32)
        # a[0,0]=focal
        # a[1,1]=focal
        # a[0,2]=W/2.
        # a[1,2]=H/2.
        if factor>1:
            a[:2]/=float(factor)
        a=np.linalg.inv(a)
        intrinsic=a
        print("intrinsic",intrinsic)

    render_poses = jt.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    # render_poses = []
    # meta = metas['test']
    # for frame in meta['frames'][:40]:
    #     render_poses.append(np.array(frame['transform_matrix']))
    # render_poses = jt.array(np.array(render_poses).astype(np.float32))
    if do_intrinsic:
        render_poses = []
        meta = metas['test']
        start = np.array(meta['frames'][0]['transform_matrix'])
        render_poses.append(start)
        for f in range(50,len(meta['frames']),50):
            end = np.array(meta['frames'][f]['transform_matrix'])
            for i in range(10):
                p=i/9.
                render_poses.append(start*(1.0-p)+end*p)
            start = end
        render_poses = jt.array(np.array(render_poses).astype(np.float32))
    
    if factor>1:
        H = H//factor
        W = W//factor
        focal = focal/float(factor)

        imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    if do_intrinsic:
        return imgs, poses, intrinsic, render_poses, [H, W, focal], i_split
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split


