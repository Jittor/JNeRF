import jittor as jt
import numpy as np

def pose_spherical(theta, phi, radius):
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
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    c2w = c2w[:-1, :]
    return c2w

def path_spherical(nframe=80):
    # poses = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,nframe+1)[:-1]]
    poses = [pose_spherical(angle, -10.0, 4.0) for angle in np.linspace(-180,180,nframe+1)[:-1]]

    # poses=[]
    # nframe=1200
    # for i in range(nframe):
    #     nt = nframe//5
    #     theta = 360/nt*(i%nt)-180
    #     phi = 90-180/nframe*i
    #     poses.append(pose_spherical(theta, phi, 4.0))

    print("poses",poses[0])
    return poses

    
# LLFF
def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses, focal):
    # hwf = poses[0, :3, -1:]
    hwf = np.expand_dims(np.array([1280, 720, focal]), -1)

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    for i in range(10):
        render_poses.append(jt.array(c2w))
    # rads = np.array(list(rads) + [1.])
    # hwf = c2w[:,4:5]
    
    # for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
    #     c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
    #     z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.]))) 
    #     render_poses.append(jt.array(np.concatenate([viewmatrix(z, up, c), hwf], 1)))
    return render_poses
    


def recenter_poses(poses, focal):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses, focal)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def path_spiral(poses, focal, nframe=10):
    print("poses",poses[0])
    poses = poses[:1]
    print("focal",focal)
    focal = focal[0][0].item()
    print("focal",focal)
    poses = poses.transpose(0,2,1).numpy()
    # poses = recenter_poses(poses, focal)
    # tmp=np.expand_dims(np.array([0,0,0,1]), axis=[0,1])
    # print("tmp",tmp.shape)
    # tmp=np.repeat(tmp,poses.shape[0],0)
    # print("tmp",tmp.shape)
    # poses = np.concatenate([poses,tmp],1)
    print("poses",poses.shape)

    # c2w = poses[0]
    c2w = poses_avg(poses, focal)
    print('recentered', c2w.shape)
    print(c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    # close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    # dt = .75
    # mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    # focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    # zdelta = close_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = nframe
    N_rots = 2

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zrate=.5, rots=N_rots, N=N_views)
    print("render_poses",len(render_poses),render_poses[0].shape)
    print("render_poses",render_poses[0])
    return render_poses