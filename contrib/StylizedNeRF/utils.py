import os
import cv2
import glob
import json
import torch
import pickle
import imageio
import plyfile
import pyrender
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib as mpl
# mpl.use('Agg')
from skimage import feature
from natsort import natsorted
import matplotlib.pyplot as plt
from plyfile import PlyElement, PlyData
from torch.utils.data import Dataset, DataLoader

import jittor as jt
from jittor import nn, Module


def json_read_rgbd(DepthImg_path, RgbImg_path, factor=1.):
    with open(DepthImg_path, 'r') as file:
        depth = np.array(json.load(file))
    rgb = Image.open(RgbImg_path).convert('RGB')
    w, h = rgb.size
    rgb = rgb.resize((int(w / factor), int(h / factor)))
    depth = cv2.resize(depth, (rgb.size[0], rgb.size[1]))
    rgb, depth = np.array(rgb, np.float32), np.array(depth, np.float32)
    return depth, rgb


def read_rgbd(DepthImg_path, RgbImg_path):
    depth_img = np.array(Image.open(DepthImg_path), np.float32)
    rgb_image = Image.open(RgbImg_path).convert('RGB')
    rgb_image = rgb_image.resize((depth_img.shape[1], depth_img.shape[0]))
    rgb_image = np.array(rgb_image, np.float32)
    return depth_img, rgb_image


def json_save_depth(path, depth):
    h, w = depth.shape[0], depth.shape[1]
    depth_list = []
    for i in range(h):
        depth_list.append(depth[i].reshape([-1]).tolist())
    with open(path, 'w') as file:
        json.dump(depth_list, file)


def write_obj(v, path, f=None):
    v = np.array(v)
    if v.shape[-1] == 3:
        str_v = [f"v {vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    else:
        str_v = [f"v {vv[0]} {vv[1]} {vv[2]} {vv[3]} {vv[4]} {vv[5]}\n" for vv in v]
    if f is not None:
        str_f = [f"f {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    else:
        str_f = []

    with open(path, 'w') as meshfile:
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


def write_ply_rgb(points, RGB, filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as PLY file """
    N = points.shape[0]
    vertex = []
    for i in range(N):
        vertex.append((points[i, 0], points[i, 1], points[i, 2], RGB[i][0], RGB[i][1], RGB[i][2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def read_ply(path):
    data = PlyData.read(path)
    coor = np.stack([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']], axis=-1)
    return coor


def read_frame_pose(path):
    """Read frame information from json file"""
    """
        Input: 
            path: json path of frame. i.e. 'frame_00000.json'
        Output:
            projectionMatrix: (4*4 ndarray) matrix of projection matrix for clipping
            intrinsic: (3*3 ndarray) intrinsic matrix of camera
            cameraPose: (4*4 ndarray) matrix of camera pose
            time: (float) time of frame
            index: (int) index of frame
    """
    with open(path, 'r') as file:
        data = json.load(file)
        projectionMatrix = np.reshape(data['projectionMatrix'], [4, 4])
        intrinsic = np.reshape(data['intrinsics'], [3, 3])
        cameraPose = np.reshape(data['cameraPoseARFrame'], [4, 4])
        time = float(data['time'])
        index = int(data['frame_index'])
    return projectionMatrix, intrinsic, cameraPose, time, index


def json_read_camera_parameters2(path, printout=False):
    with open(path, 'r') as file:
        data = json.load(file)
        timeStamp = data['timeStamp']
        cameraEulerAngle = data['cameraEulerAngle']
        imageResolution = data['imageResolution']
        cameraTransform = np.reshape(data['cameraTransform'], [4, 4])
        cameraPos = data['cameraPos']
        cameraIntrinsics = np.reshape(data['cameraIntrinsics'], [3, 3])
        cameraView = np.reshape(data['cameraView'], [4, 4])
        cameraProjection = np.reshape(data['cameraProjection'], [4, 4])

    if printout:
        parameters = [timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection]
        names = ['timeStamp', 'cameraEulerAngle', 'imageResolution', 'cameraTransform', 'cameraPos', 'cameraIntrinsics', 'cameraView', 'cameraProjection']
        for i in range(len(parameters)):
            print('******************************************************************************************')
            print(names[i])
            print(parameters[i])
            print('******************************************************************************************')

    return timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection


def json_read_camera_parameters(path, printout=False):
    with open(path, 'r') as file:
        data = json.load(file)
        timeStamp = []
        cameraEulerAngle = []
        imageResolution = []
        cameraTransform = np.reshape(data['cameraTransform'], [4, 4])
        cameraPos = []
        cameraIntrinsics = np.reshape(data['cameraIntrinsics'], [3, 3])
        cameraView = []
        cameraProjection = []

    if printout:
        parameters = [timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection]
        names = ['timeStamp', 'cameraEulerAngle', 'imageResolution', 'cameraTransform', 'cameraPos', 'cameraIntrinsics', 'cameraView', 'cameraProjection']
        for i in range(len(parameters)):
            print('******************************************************************************************')
            print(names[i])
            print(parameters[i])
            print('******************************************************************************************')

    return timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection


def json_save_camera_parameters(path, cp, intr):
    timeStamp = []
    cameraEulerAngle = []
    imageResolution = []
    cameraTransform = np.reshape(cp, [-1]).tolist()
    cameraPos = []
    cameraIntrinsics = np.reshape(intr, [-1]).tolist()
    cameraView = []
    cameraProjection = []

    parameters = [timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection]
    names = ['timeStamp', 'cameraEulerAngle', 'imageResolution', 'cameraTransform', 'cameraPos', 'cameraIntrinsics', 'cameraView', 'cameraProjection']
    save_dict = {}
    for i in range(len(parameters)):
            save_dict[names[i]] = parameters[i]
    with open(path, 'w') as file:
        json.dump(save_dict, file)


def write_ply(v, path):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n"
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]

    with open(path, 'w') as meshfile:
        meshfile.write(f'{header}{"".join(str_v)}')


def load_ply(path):
    data = plyfile.PlyData.read(path)
    pcls = np.array([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']], np.float32).T
    rgbs = np.array([data['vertex']['red'], data['vertex']['green'], data['vertex']['blue']], np.float32).T
    return pcls, rgbs


def save_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def view_synthesis(cps, factor=10):
    frame_num = cps.shape[0]
    cps = np.array(cps)
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R
    from scipy import interpolate as intp
    rots = R.from_matrix(cps[:, :3, :3])
    slerp = Slerp(np.arange(frame_num), rots)
    tran = cps[:, :3, -1]
    f_tran = intp.interp1d(np.arange(frame_num), tran.T)

    new_num = int(frame_num * factor)

    new_rots = slerp(np.linspace(0, frame_num - 1, new_num)).as_matrix()
    new_trans = f_tran(np.linspace(0, frame_num - 1, new_num)).T

    new_cps = np.zeros([new_num, 4, 4], np.float)
    new_cps[:, :3, :3] = new_rots
    new_cps[:, :3, -1] = new_trans
    new_cps[:, 3, 3] = 1
    return new_cps


def normalize_cps(cps):
    cps = np.array(cps, dtype=np.float32)
    avg_center = min_line_dist_center(cps[:, :3, 3], cps[:, :3, 2])
    cps[:, :3, 3] -= avg_center
    dists = np.linalg.norm(cps[:, :3, 3], axis=-1)
    radius = 1.1 * np.max(dists) + 1e-5
    # Corresponding parameters change
    cps[:, :3, 3] /= radius
    return cps, radius


def min_line_dist_center(rays_o, rays_d):
    if len(np.shape(rays_d)) == 2:
        rays_o = rays_o[..., np.newaxis]
        rays_d = rays_d[..., np.newaxis]
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((A_i @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist


def save_obj(path, obj):
    file = open(path, 'wb')
    obj_str = pickle.dumps(obj)
    file.write(obj_str)
    file.close()


def load_obj(path):
    file = open(path, 'rb')
    obj = pickle.loads(file.read())
    file.close()
    return obj


class plot_chart:
    def __init__(self, name='image', path='./plotting/', x_label='iter', y_label='Loss', max_len=100000):
        self.name = name
        self.path = path
        self.x_label = x_label
        self.y_label = y_label
        self.max_len = max_len
        self.ys, self.xs = None, None
        self.path = './chart'

    def draw(self, y, x):
        self.ys = np.array([y]) if self.ys is None else np.concatenate([self.ys, [y]])
        self.xs = np.array([x]) if self.xs is None else np.concatenate([self.xs, [x]])

        self.check_len()

        plt.close('all')
        plt.plot(self.xs, self.ys, "b.-")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        plt.savefig(self.path + "/" + self.name + ".png")

        self.save()

    def check_len(self):
        if self.ys.shape[0] > self.max_len:
            half_ids = np.arange(self.ys.shape[0]//2, self.ys.shape[0])
            self.ys = self.ys[half_ids]
            self.xs = self.xs[half_ids]

    def save(self):
        save_obj(self.path + '/chart_obj', self)


def get_rays_ios_np(H, W, focal, c2w, cx=None, cy=None):
    if cx is None or cy is None:
        cx, cy = W * .5, H * .5
    # else:
    #     print("Cx from %.03f to %.03f, Cy from %.03f to %.03f" % (H/2, cx, W/2, cy))
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/focal, -(j-cy)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_from_id(hid, wid, focal, c2w, cx, cy):
    dir = np.stack([(wid - cx) / focal, - (hid - cy) / focal, -np.ones_like(wid)], axis=-1)
    ray_d = np.einsum('wc,c->w', c2w[:3, :3], dir)
    ray_d = ray_d / np.linalg.norm(ray_d)
    ray_o = c2w[:3, -1]
    ray_o, ray_d = np.array(ray_o, dtype=np.float32), np.array(ray_d, dtype=np.float32)
    return ray_o, ray_d


def dep2pcl(depth, intr, cp, pixel_alignment=True):
    intr = intr.copy()
    h, w = np.shape(depth)[:2]
    if pixel_alignment:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32) - 0.5, np.arange(h, dtype=np.float32) - 0.5, indexing='xy')
    else:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    z = - depth
    uvz = np.stack([u*z, v*z, z], axis=-1).reshape([-1, 3])
    # The z axis is toward the camera and y axis should be conversed
    intr[0, 0] = - intr[0, 0]
    intr_inverse = np.linalg.inv(intr)
    xyz_camera = np.einsum('bu,cu->bc', uvz, intr_inverse)
    xyz_camera = np.concatenate([xyz_camera, np.ones([xyz_camera.shape[0], 1])], axis=-1)
    xyz_world = np.einsum('bc,wc->bw', xyz_camera, cp)
    return xyz_world


def get_cos_map(h, w, cx, cy, f):
    i, j = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/f, -(j-cy)/f, -np.ones_like(i)], -1)
    cos = 1 / np.linalg.norm(dirs, axis=-1)
    cos = np.array(cos, dtype=np.float32)
    return cos


def pts2imgcoor(pts, intr):
    intr = intr.copy()
    intr[0, 0] *= -1
    imgcoor = np.einsum('bc,ic->bi', pts, intr)
    imgcoor /= imgcoor[..., -1][..., np.newaxis]
    return imgcoor


def alpha_composition(pts_rgb, pts_sigma, t_values, sigma_noise_std=0., white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        pts_rgb: [num_rays, num_samples along ray, 3]. Prediction from model.
        pts_sigma: [num_rays, num_samples along ray]. Prediction from model.
        t_values: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_exp: [num_rays, 3]. Estimated RGB color of a ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        t_exp: [num_rays]. Estimated distance to object.
    """
    # sigma2alpha = lambda sigma, dists: 1.-jt.exp(-sigma * dists)
    sigma2alpha = lambda raw, dists, act_fn=jt.nn.relu: 1. - jt.exp(-act_fn(raw) * dists)

    delta = t_values[..., 1:] - t_values[..., :-1]
    delta = jt.concat([delta, jt.array([1e10]).expand(delta[..., :1].shape)], -1)  # [N_rays, N_samples]

    noise = 0.
    if sigma_noise_std > 0:
        # noise = jt.random(pts_sigma.shape) * sigma_noise_std
        noise = jt.init.gauss(pts_sigma.shape, pts_sigma.dtype) * sigma_noise_std

    alpha = sigma2alpha(jt.nn.relu(pts_sigma + noise), delta)  # [N_rays, N_samples]
    weights = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_exp = jt.sum(weights[..., None] * pts_rgb, -2)  # [N_rays, 3]

    t_exp = jt.sum(weights * t_values, -1)
    acc_map = jt.sum(weights, -1)
    if white_bkgd:
        rgb_exp = rgb_exp + (1. - acc_map[..., None])

    return rgb_exp, t_exp, weights


def alpha_composition_wild(pts_rgb, pts_sigma, t_values, pts_transient_rgb, pts_transient_sigma, pts_transient_beta, beta_min=0.03, sigma_noise_std=0., white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        pts_rgb: [num_rays, num_samples along ray, 3]. Prediction from model.
        pts_sigma: [num_rays, num_samples along ray]. Prediction from model.
        t_values: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_exp: [num_rays, 3]. Estimated RGB color of a ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        t_exp: [num_rays]. Estimated distance to object.
    """
    sigma2alpha = lambda sigma, dists: 1.-torch.exp(-sigma * dists)

    delta = t_values[..., 1:] - t_values[..., :-1]
    delta = torch.cat([delta, torch.Tensor([1e10]).expand(delta[..., :1].shape).to(pts_rgb.device)], -1)  # [N_rays, N_samples]

    noise = 0.
    if sigma_noise_std > 0:
        noise = torch.randn(pts_sigma.shape, device=pts_sigma.device) * sigma_noise_std

    sigma_static = torch.relu(pts_sigma + noise)
    alpha_static = sigma2alpha(sigma_static, delta)

    sigma_transient = torch.relu(pts_transient_sigma)
    alpha_transient = sigma2alpha(sigma_transient, delta)
    T_transient = torch.cumprod(torch.cat([torch.ones((alpha_transient.shape[0], 1), device=alpha_transient.device), 1. - alpha_transient + 1e-10], -1), -1)[:, :-1]
    beta_exp = torch.sum(T_transient[..., None] * alpha_transient[..., None] * torch.relu(pts_transient_beta), -2) + beta_min

    sigma_both = sigma_static + sigma_transient
    alpha_both = sigma2alpha(sigma_both, delta)  # [N_rays, N_samples]
    T_both = torch.cumprod(torch.cat([torch.ones((alpha_both.shape[0], 1), device=alpha_both.device), 1.-alpha_both + 1e-10], -1), -1)[:, :-1]

    rgb_exp = torch.sum(T_both[..., None] * alpha_static[..., None] * pts_rgb + T_both[..., None] * alpha_transient[..., None] * pts_transient_rgb, -2)  # [N_rays, 3]

    weights = alpha_both * T_both
    t_exp = torch.sum(weights * t_values, -1)
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_exp = rgb_exp + (1.-acc_map[..., None])

    return rgb_exp, t_exp, weights, beta_exp


def batchify(fn, chunk=1024*32):
    """Render rays in smaller minibatches to avoid OOM.
    """
    if chunk is None:
        return fn

    def ret_func(**kwargs):
        x = kwargs[list(kwargs.keys())[0]]
        all_ret = {}
        for i in range(0, x.shape[0], chunk):
            end = min(i + chunk, x.shape[0])
            chunk_kwargs = dict([[key, kwargs[key][i: end]] for key in kwargs.keys()])
            ret = fn(**chunk_kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: jt.concat(all_ret[k], 0) for k in all_ret}
        return all_ret

    return ret_func


img2mse = lambda x, y: jt.mean((x - y) ** 2)
img2l1 = lambda x, y: (x - y).abs().mean()
mse2psnr = lambda x: -10. * jt.log(x) / jt.log(jt.array([10.]))
to8b = lambda x: np.array(x, dtype=np.uint8)


def get_rays(H, W, focal, cps, cx=None, cy=None, chunk=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = int(H), int(W)
    if cx is None or cy is None:
        cx, cy = W * .5, H * .5
    j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
    dirs = torch.stack([(i-cx)/focal, -(j-cy)/focal, -torch.ones_like(i)], -1).to(device)
    cps_tensor = torch.from_numpy(cps).float().to(device)
    start = 0
    rays_o_total, rays_d_total = np.zeros([cps.shape[0], H, W, 3], np.float32), np.zeros([cps.shape[0], H, W, 3], np.float32)
    while start < cps.shape[0]:
        print('\rProcess: %.3f%%' % (start / cps.shape[0] * 100), end='')
        end = min(start + chunk, cps.shape[0])

        rays_d = torch.einsum('hwc,nbc->nhwb', dirs, cps_tensor[start: end, :3, :3])
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_d = rays_d.cpu().numpy()
        rays_o = np.broadcast_to(cps[start: end, :3, -1][:, np.newaxis, np.newaxis], np.shape(rays_d))

        rays_o_total[start: end] = rays_o
        rays_d_total[start: end] = rays_d
        start = end
    print('\rProcess: 100.000%%')

    return rays_o_total, rays_d_total


def empty_loss(ts, sigma, t_gt):
    """Empty Loss"""
    """
    ts: [ray, N]
    sigma: [ray, N]
    t_gt: [ray]
    """
    delta_ts = ts[:, 1:] - ts[:, :-1]  # [ray, N-1]
    sigma = torch.relu(sigma[:, :-1])  # [ray, N-1]
    boarder_rate = 0.9
    sigma_sum = torch.sum(sigma * delta_ts * (ts[:, :-1] < (t_gt.unsqueeze(-1) * boarder_rate)).float(), dim=-1)
    loss_empty = torch.mean(sigma_sum)
    return loss_empty


def sampling_pts_uniform(rays_o, rays_d, N_samples=64, near=0., far=1.05, harmony=False, perturb=False):
    #  Intersect, ts_nf of shape [ray, box] and [ray, box, 2]
    ray_num = rays_o.shape[0]

    #  Uniform sampling ts of shape [ray, N_samples]
    ts = jt.linspace(0, 1, N_samples).unsqueeze(0).expand(ray_num, N_samples)
    if not harmony:
        ts = ts * (far - near) + near
    else:
        ts = 1. / (1./near * (1 - ts) + 1./far * ts)

    if perturb:
        #  Add perturb
        rand = jt.zeros([ray_num, N_samples])
        jt.init.uniform_(rand, 0, 1)
        mid = (ts[..., 1:] + ts[..., :-1]) / 2
        upper = jt.concat([mid, ts[..., -1:]], -1)
        lower = jt.concat([ts[..., :1], mid], -1)
        ts = lower + (upper - lower) * rand

    #  From ts to pts. [ray, N_samples, 3]
    rays_o, rays_d = rays_o.unsqueeze(1).expand([ray_num, N_samples, 3]), rays_d.unsqueeze(1).expand([ray_num, N_samples, 3])
    ts_expand = ts.unsqueeze(-1).expand([ray_num, N_samples, 3])
    pts = rays_o + ts_expand * rays_d

    return pts, ts


def sampling_pts_fine(rays_o, rays_d, ts, weights, N_samples_fine=64):

    ray_num, N_samples = ts.shape
    # ts of shape [ray, N_samples], ts_mid of shape [ray, N_samples - 1]
    ts_mid = 0.5 * (ts[..., 1:] + ts[..., :-1])
    # pdf of shape [ray, N_samples - 2]
    weights = weights[..., 1:-1] + 1e-3
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    # cdf of shape [ray, N_samples - 1]
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)
    # random sampling of shape [ray, N_samples_fine]
    u = jt.random(list(cdf.shape[:-1]) + [N_samples_fine]) * (1-1e-3)  # Avoid sample 1
    # inds below of shape [ray, N_samples_fine] in range [0, N_samples - 3]
    below = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(below - 1), below - 1)
    below = jt.minimum((N_samples - 3) * jt.ones_like(below), below)
    # Use below to gather cdf. [ray, N_samples_fine]
    ray_Nfine_N_1 = [ray_num, N_samples_fine, N_samples - 1]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(ray_Nfine_N_1), -1, below.unsqueeze(-1)).squeeze(-1)
    # Interval t values of cdf (pdf). [ray, N_samples_fine]
    ray_Nfine_N_2 = [ray_num, N_samples_fine, N_samples - 2]
    pdf_g = jt.gather(pdf.unsqueeze(1).expand(ray_Nfine_N_2), -1, below.unsqueeze(-1)).squeeze(-1)
    pdf_g = jt.ternary(pdf_g == 0, jt.ones_like(pdf_g), pdf_g)
    # ts in each interval. [ray, N_samples_fine]
    ts_interval = (u - cdf_g) / pdf_g
    # Above index of shape [ray, N_samples_fine] in range(1, N_samples - 2)
    above = jt.minimum((cdf.shape[-1] - 1) * jt.ones_like(below), below + 1)
    # ts boarder of each interval. [ray, N_samples_fine]
    ts_near = jt.gather(ts_mid.unsqueeze(1).expand([ray_num, N_samples_fine, N_samples-1]), -1, below.unsqueeze(-1)).squeeze(-1)
    ts_far = jt.gather(ts_mid.unsqueeze(1).expand([ray_num, N_samples_fine, N_samples-1]), -1, above.unsqueeze(-1)).squeeze(-1)
    # ts_fine of shape [ray, N_samples_fine]
    ts_fine = ts_near + ts_interval * (ts_far - ts_near)
    # Sort from near to far [ray, N_samples + N_samples_fine]
    ts = jt.concat([ts, ts_fine], dim=-1)
    _, ts = jt.argsort(ts, dim=-1)
    # Avoid BP
    ts = ts.detach()

    #  From ts to pts. [ray, N_samples + N_samples_fine, 3]
    rays_o, rays_d = rays_o.unsqueeze(1).expand([ray_num, N_samples + N_samples_fine, 3]), rays_d.unsqueeze(1).expand([ray_num, N_samples + N_samples_fine, 3])
    ts_expand = ts.unsqueeze(-1).expand([ray_num, N_samples + N_samples_fine, 3])
    pts = rays_o + ts_expand * rays_d
    pts = pts.detach()

    return pts, ts


def sampling_pts_fine_jt(rays_o, rays_d, ts, weights, N_samples_fine=64):

    # ts of shape [ray, N_samples], ts_mid of shape [ray, N_samples - 1]
    ts_mid = 0.5 * (ts[..., 1:] + ts[..., :-1])
    t_samples = sample_pdf(ts_mid, weights[..., 1:-1], N_samples_fine, det=True)
    t_samples = t_samples.detach()
    _, t_vals = jt.argsort(jt.concat([ts, t_samples], -1), -1)
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)  # [N_rays, N_samples + N_importance, 3]

    # Avoid BP
    t_vals = t_vals.detach()

    return pts, t_vals


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.random(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    cond = jt.where(denom < 1e-5)
    denom[cond[0], cond[1]] = 1.
    t = (u-cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples

