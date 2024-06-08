import torch
import numpy as np

import torch.nn as torch_nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

import platform

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

import jittor as jt
from jittor import Module
from jittor import nn


class Sine(Module):
    def __init(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return jt.sin(self.w0 * input)


act_dict = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU, 'tanh': nn.Tanh, 'sine': Sine}


class Embedder(Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(jt.sin, jt.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** np.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = np.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.tolist()

    def execute(self, x):
        '''
        :param x: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (x.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(x)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(x * freq))
        out = jt.concat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


class MLP(Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3, skips=[4], act_func=nn.ReLU, use_viewdir=True,
                 sigma_mul=0.):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        '''
        super().__init__()
        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.skips = skips
        self.use_viewdir = use_viewdir
        self.sigma_mul = sigma_mul

        # base
        self.base_layers = []
        dim = self.input_ch
        for i in range(D):
            self.base_layers.append(nn.Linear(in_features=dim, out_features=W, bias=True))
            dim = W
            if i in self.skips and i != (D - 1):  # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)
        self.act = act_func()

        # sigma
        sigma_layer = nn.Linear(dim, 1)  # sigma must be positive
        self.sigma_layer = sigma_layer

        # remap
        base_remap_layer = nn.Linear(dim, 256)
        self.base_remap_layer = base_remap_layer

        # rgb
        self.rgb_layers = []
        dim = 256 + self.input_ch_viewdirs if self.use_viewdir else 256
        self.rgb_layers.append(nn.Linear(dim, W // 2))
        self.rgb_layers.append(nn.Linear(W // 2, 3))
        self.rgb_layers = nn.ModuleList(self.rgb_layers)

        self.layers = [*self.base_layers, self.sigma_layer, self.base_remap_layer, *self.rgb_layers]

    def execute(self, pts, dirs):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        base = self.base_layers[0](pts)
        for i in range(len(self.base_layers) - 1):
            if i in self.skips:
                base = torch.cat((pts, base), dim=-1)
            base = self.act(self.base_layers[i + 1](base))

        sigma = self.sigma_layer(base)
        sigma = sigma + nn.relu(sigma) * self.sigma_mul

        base_remap = self.act(self.base_remap_layer(base))
        if self.use_viewdir:
            rgb_fea = self.act(self.rgb_layers[0](torch.cat((base_remap, dirs), dim=-1)))
        else:
            rgb_fea = self.act(self.rgb_layers[0](base_remap))
        rgb = jt.sigmoid(self.rgb_layers[1](rgb_fea))

        ret = OrderedDict([('rgb', rgb),
                           ('sigma', sigma.squeeze(-1))])
        return ret

    def get_grads(self, only_last=False):
        if only_last:
            layers = [self.layers[-1], self.layers[-4]]
        else:
            layers = self.layers
        grads = None
        for layer in layers:
            grad = layer.get_grads()
            grads = grad if grads is None else np.concatenate([grads, grad], axis=-1)
        return grads


class MLP_style(Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3, skips=[4], act_func=nn.ReLU(), use_viewdir=True,
                 sigma_mul=0., enable_style=False):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        '''
        super().__init__()
        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.skips = skips
        self.use_viewdir = use_viewdir
        self.sigma_mul = sigma_mul
        self.enable_style = enable_style
        self.act = act_func()

        # base
        self.base_layers = []
        dim = self.input_ch
        for i in range(D):
            self.base_layers.append(nn.Linear(in_features=dim, out_features=W))
            dim = W
            if i in self.skips and i != (D - 1):  # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)

        # sigma
        sigma_layer = nn.Linear(dim, 1)  # sigma must be positive
        self.sigma_layer = sigma_layer

        # remap
        base_remap_layer = nn.Linear(dim, 256)
        self.base_remap_layer = base_remap_layer

        # rgb
        self.rgb_layers = []
        dim = 256 + self.input_ch_viewdirs if self.use_viewdir else 256
        self.rgb_layers.append(nn.Linear(dim, W // 2))
        self.rgb_layers.append(nn.Linear(W // 2, 3))
        self.rgb_layers = nn.ModuleList(self.rgb_layers)

        self.layers = [*self.base_layers, self.sigma_layer, self.base_remap_layer, *self.rgb_layers]

    def execute(self, **kwargs):
        pts, dirs = kwargs['pts'], kwargs['dirs']
        base = self.act(self.base_layers[0](pts))
        for i in range(len(self.base_layers) - 1):
            if i in self.skips:
                base = jt.concat((pts, base), dim=-1)
            base = self.act(self.base_layers[i + 1](base))

        sigma = self.sigma_layer(base)
        sigma = sigma + jt.nn.relu(sigma) * self.sigma_mul

        base_remap = self.act(self.base_remap_layer(base))
        if self.use_viewdir:
            rgb_fea = self.act(self.rgb_layers[0](jt.concat((base_remap, dirs), dim=-1)))
        else:
            rgb_fea = self.act(self.rgb_layers[0](base_remap))
        rgb = jt.sigmoid(self.rgb_layers[1](rgb_fea))

        if self.enable_style:
            ret = OrderedDict([('rgb', rgb),
                               # ('base', base),  # for base input style nerf
                               ('pts', pts),
                               ('sigma', sigma.squeeze(-1))])
            return ret
        else:
            ret = OrderedDict([('rgb', rgb),
                               ('sigma', sigma.squeeze(-1))])
            return ret


class Nerf(Module):
    def __init__(self, args, mode='coarse'):
        super().__init__()
        self.use_viewdir = args.use_viewdir
        """Activation Function"""
        act_func = act_dict[args.act_type]
        self.is_siren = (args.act_type == 'sine')

        """Embedding"""
        if not self.is_siren:
            self.embedder_coor = Embedder(input_dim=3, max_freq_log2=args.embed_freq_coor - 1,
                                          N_freqs=args.embed_freq_coor)
            self.embedder_dir = Embedder(input_dim=3, max_freq_log2=args.embed_freq_dir - 1,
                                         N_freqs=args.embed_freq_dir)
            input_ch, input_ch_viewdirs = self.embedder_coor.out_dim, self.embedder_dir.out_dim
            skips = [4]
            self.sigma_mul = 0.
        else:
            input_ch, input_ch_viewdirs = 3, 3
            skips = []
            self.sigma_mul = args.siren_sigma_mul

        """Neural Network"""
        if mode == 'coarse':
            net_depth, net_width = args.netdepth, args.netwidth
        else:
            net_depth, net_width = args.netdepth_fine, args.netwidth_fine

        self.net = MLP(D=net_depth, W=net_width, input_ch=input_ch, input_ch_viewdirs=input_ch_viewdirs,
                       skips=skips, use_viewdir=self.use_viewdir, act_func=act_func, sigma_mul=self.sigma_mul)

    def execute(self, pts, dirs):
        if not self.is_siren:
            pts = self.embedder_coor(pts)
            dirs = self.embedder_dir(dirs)
        ret = self.net(pts, dirs)
        return ret


class StyleMLP(Module):
    def __init__(self, args):
        super().__init__()
        self.D = args.style_D
        self.input_ch = args.embed_freq_coor * 3 * 2 + 3 + args.vae_latent
        self.layers = []
        self.skips = [4]
        dim = self.input_ch
        for i in range(self.D-1):
            if i in self.skips:
                dim += self.input_ch
            self.layers.append(nn.Linear(dim, args.netwidth))
            dim = args.netwidth
        self.layers.append(nn.Linear(args.netwidth, 3))
        self.layers = nn.ModuleList(self.layers)

    def execute(self, **kwargs):
        x = kwargs['x']
        h = x
        for i in range(len(self.layers)-1):
            if i in self.skips:
                h = jt.concat([h, x], dim=-1)
            h = self.layers[i](h)
            h = nn.relu(h)
        h = self.layers[-1](h)
        h = jt.sigmoid(h)
        return {'rgb': h}


class StyleMLP_Wild_multilayers(Module):
    def __init__(self, args):
        super().__init__()
        self.D = args.style_D
        self.input_ch = args.embed_freq_coor * 3 * 2 + 3 + args.vae_latent
        self.layers = []
        self.skips = [4]
        dim = self.input_ch
        for i in range(self.D-1):
            if i in self.skips:
                dim += (args.embed_freq_coor * 3 * 2 + 3)
            self.layers.append(nn.Linear(dim, args.netwidth))
            dim = args.netwidth + args.vae_latent
        self.layers.append(nn.Linear(args.netwidth + args.vae_latent, 3))
        self.layers = nn.ModuleList(self.layers)

    def execute(self, **kwargs):
        x = kwargs['x']
        l = kwargs['latent']
        h = x
        for i in range(len(self.layers)-1):
            h = jt.concat([h, l], dim=-1)
            if i in self.skips:
                h = jt.concat([h, x], dim=-1)
            h = self.layers[i](h)
            h = nn.relu(h)
        h = jt.concat([h, l], dim=-1)
        h = self.layers[-1](h)
        h = jt.sigmoid(h)
        return {'rgb': h}


class StyleNerf(Module):
    def __init__(self, args, mode='coarse', enable_style=False):
        super().__init__()
        self.use_viewdir = args.use_viewdir
        """Activation Function"""
        act_func = act_dict[args.act_type]
        self.is_siren = (args.act_type == 'sine')

        """Embedding"""
        if not self.is_siren:
            self.embedder_coor = Embedder(input_dim=3, max_freq_log2=args.embed_freq_coor - 1,
                                          N_freqs=args.embed_freq_coor)
            self.embedder_dir = Embedder(input_dim=3, max_freq_log2=args.embed_freq_dir - 1,
                                         N_freqs=args.embed_freq_dir)
            input_ch, input_ch_viewdirs = self.embedder_coor.out_dim, self.embedder_dir.out_dim
            skips = [4]
            self.sigma_mul = 0.
        else:
            input_ch, input_ch_viewdirs = 3, 3
            skips = []
            self.sigma_mul = args.siren_sigma_mul

        """Neural Network"""
        if mode == 'coarse':
            net_depth, net_width = args.netdepth, args.netwidth
        else:
            net_depth, net_width = args.netdepth_fine, args.netwidth_fine

        self.net = MLP_style(D=net_depth, W=net_width, input_ch=input_ch, input_ch_viewdirs=input_ch_viewdirs,
                             skips=skips, use_viewdir=self.use_viewdir, act_func=act_func, sigma_mul=self.sigma_mul, enable_style=enable_style)
        self.enable_style = enable_style

    def set_enable_style(self, enable_style=False):
        self.enable_style = enable_style
        self.net.enable_style = enable_style

    def execute(self, **kwargs):
        # mode consistency
        self.net.enable_style = self.enable_style
        if not self.is_siren:
            kwargs['pts'] = self.embedder_coor(kwargs['pts'])
            kwargs['dirs'] = self.embedder_dir(kwargs['dirs'])
        ret = self.net(**kwargs)
        ret['dirs'] = kwargs['dirs']
        return ret


def vec2skew(v):
    """
    :param v:  (N, 3, ) torch tensor
    :return:   (N, 3, 3)
    """
    zero = jt.zeros([v.shape[0], 1], dtype=jt.float32, device=v.device)
    skew_v0 = jt.concat([zero, -v[:, 2:3], v[:, 1:2]], dim=-1)  # (N, 3)
    skew_v1 = jt.concat([v[:, 2:3], zero, -v[:, 0:1]], dim=-1)
    skew_v2 = jt.concat([-v[:, 1:2], v[:, 0:1], zero], dim=-1)
    skew_v = jt.stack([skew_v0, skew_v1, skew_v2], dim=-1)  # (N, 3, 3)
    return skew_v


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (N, 3) axis-angle, torch tensor
    :return:  (N, 3, 3)
    """
    skew_r = vec2skew(r)  # (N, 3, 3)
    norm_r = r.norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-15  # (N, 1, 1)
    eye = jt.init.eye(3).unsqueeze(0)  # (1, 3, 3)
    R = eye + (jt.sin(norm_r) / norm_r) * skew_r + ((1 - jt.cos(norm_r)) / norm_r ** 2) * jt.matmul(skew_r, skew_r)
    return R


def make_c2w(r, t):
    """
    :param r:  (N, 3, ) axis-angle             torch tensor
    :param t:  (N, 3, ) translation vector     torch tensor
    :return:   (N, 4, 4)
    """
    R = Exp(r)  # (N, 3, 3)
    c2w = jt.concat([R, t.unsqueeze(-1)], dim=-1)  # (N, 3, 4)
    c2w = jt.concat([c2w, jt.zeros_like(c2w[:, :1])], dim=1)  # (N, 4, 4)
    c2w[:, 3, 3] = 1.
    return c2w


def idx2img(idx, fea, pad=0):
    batch_size, h, w, z = idx.shape
    batch_size_p, point_num, dim = fea.shape
    assert batch_size == batch_size_p, 'Batch Size Do Not Match'
    idx_img = idx.reshape([batch_size, h*w*z, 1]).expand([batch_size, h*w*z, dim]).long()
    idx_lst = point_num * torch.ones_like(idx_img)
    idx_img = torch.where(idx_img >= 0, idx_img, idx_lst)
    fea_pad = fea.reshape([1, batch_size*point_num, dim]).expand([batch_size, batch_size*point_num, dim])
    fea_pad = torch.cat([fea_pad, pad * torch.ones([batch_size, 1, dim]).to(idx.device)], dim=1)
    fea_img = torch.gather(fea_pad, 1, idx_img).reshape([batch_size, h, w, z, dim])
    return fea_img


class Camera:
    def __init__(self, projectionMatrix=None, cameraPose=None, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.tensor_list = ['projectionMatrix', 'cameraPose', 'w2c_matrix']
        for attr in self.tensor_list:
            setattr(self, attr, None)
        self.set(projectionMatrix=projectionMatrix, cameraPose=cameraPose)

    def set(self, **kwargs):
        keys = kwargs.keys()
        func_map = {'projectionMatrix': self.set_project, 'cameraPose': self.set_pose}
        for name in keys:
            try:
                if name in func_map.keys():
                    func_map[name](kwargs[name])
                else:
                    raise ValueError(name + f'is not in{keys}')
            except ValueError as e:
                print(repr(e))

    def set_pose(self, cameraPose):
        if cameraPose is None:
            self.cameraPose = self.w2c_matrix = None
            return
        elif type(cameraPose) is np.ndarray:
            cameraPose = torch.from_numpy(cameraPose)
        self.cameraPose = cameraPose.float()
        self.w2c_matrix = torch.inverse(self.cameraPose).float()
        self.to(self.device)

    def set_project(self, projectionMatrix):
        if projectionMatrix is None:
            self.projectionMatrix = None
            return
        elif type(projectionMatrix) is np.ndarray:
            projectionMatrix = torch.from_numpy(projectionMatrix)
        self.projectionMatrix = projectionMatrix.float()
        self.to(self.device)

    def to(self, device):
        if type(device) is str:
            device = torch.device(device)
        self.device = device
        for tensor in self.tensor_list:
            if getattr(self, tensor) is not None:
                setattr(self, tensor, getattr(self, tensor).to(self.device))
        return self

    def WorldtoCamera(self, coor_world):
        coor_world = coor_world.clone()
        if len(coor_world.shape) == 2:
            coor_world = torch.cat([coor_world, torch.ones([coor_world.shape[0], 1]).to(self.device)], -1)
            coor_camera = torch.einsum('bcw,nw->bnc', self.w2c_matrix, coor_world)
        else:
            coor_world = self.homogeneous(coor_world)
            coor_camera = torch.einsum('bcw,bnw->bnc', self.w2c_matrix, coor_world)
        return coor_camera

    def CameratoWorld(self, coor_camera):
        coor_camera = coor_camera.clone()
        coor_camera = self.homogeneous(coor_camera)
        coor_world = torch.einsum('bwc,bnc->bnw', self.cameraPose, coor_camera)[:, :, :3]
        return coor_world

    def WorldtoCVV(self, coor_world):
        coor_camera = self.WorldtoCamera(coor_world)
        coor_cvv = torch.einsum('vc,bnc->bnv', self.projectionMatrix, coor_camera)
        coor_cvv = coor_cvv[..., :-1] / coor_cvv[..., -1:]
        return coor_cvv

    def homogeneous(self, coor3d, force=False):
        if coor3d.shape[-1] == 3 or force:
            coor3d = torch.cat([coor3d, torch.ones_like(coor3d[..., :1]).to(self.device)], -1)
        return coor3d

    def rasterize(self, coor_world, rgb, h=192, w=256, k=1.5, z=1):
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer import compositing
        from pytorch3d.renderer.points import rasterize_points

        def PixeltoCvv(h, w, hid=0, wid=0):
            cvv = torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]]).float()
            pts = Pointclouds(points=cvv, features=cvv)
            idx, _, dist2 = rasterize_points(pts, [h, w], 1e10, 3)
            a2, b2, c2 = (dist2.cpu().numpy())[0, hid, wid]
            x2 = (a2 + b2) / 2 - 1
            cosa = (x2 + 1 - a2) / (2 * x2**0.5)
            sina_abs = (1 - cosa**2)**0.5
            u = (x2 ** 0.5) * cosa
            v = (x2 ** 0.5) * sina_abs
            if np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5) > 1e-5:
                v = - (x2 ** 0.5) * sina_abs
                if(np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5) > 1e-5):
                    print(np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5), ' is too large...')
                    print(f"Found pixel {[hid, wid]} has uv: {(u, v)} But something wrong !!!")
                    print(f"a: {a2**0.5}, b: {b2**0.5}, c: {c2**0.5}, idx: {idx[0, 0, 0]}, dist2: {dist2[0, 0, 0]}")
                    os.exit(-1)
            return u, v

        batch_size = self.cameraPose.shape[0]
        point_num = rgb.shape[-2]
        coor_cvv = self.WorldtoCVV(coor_world).reshape([batch_size, point_num, 3])  # (batch_size, point, 3)
        umax, vmax = PixeltoCvv(h=h, w=w, hid=0, wid=0)
        umin, vmin = PixeltoCvv(h=h, w=w, hid=h-1, wid=w-1)
        cvv_backup = coor_cvv.clone()
        coor_cvv[..., 0] = (coor_cvv[..., 0] + 1) / 2 * (umax - umin) + umin
        coor_cvv[..., 1] = (coor_cvv[..., 1] + 1) / 2 * (vmax - vmin) + vmin

        rgb = rgb.reshape([1, point_num, rgb.shape[-1]])  # (1, point, 3)
        rgb_coor = torch.cat([rgb, coor_world.unsqueeze(0)], dim=-1).expand([batch_size, point_num, 6])  # (1, point, 6)

        if platform.system() == 'Windows':
            # Bug of pytorch3D on windows
            hw = np.array([h, w])
            mindim, maxdim = np.argmin(hw), np.argmax(hw)
            aspect_ration = hw[maxdim] / hw[mindim]
            coor_cvv[:, :, mindim] *= aspect_ration

        pts3D = Pointclouds(points=coor_cvv, features=rgb_coor)
        radius = float(2. / max(w, h) * k)
        idx, _, _ = rasterize_points(pts3D, [h, w], radius, z)
        alphas = torch.ones_like(idx.float())
        img = compositing.alpha_composite(
            idx.permute(0, 3, 1, 2).long(),
            alphas.permute(0, 3, 1, 2),
            pts3D.features_packed().permute(1, 0),
        )
        img = img.permute([0, 2, 3, 1]).contiguous()  # (batch, h, w, 6)
        rgb_map, coor_map = img[..., :3], img[..., 3:]  # (batch, h, w, 3)
        msk = (idx[:, :, :, :1] != -1).float()  # (batch, h, w, 1)

        return rgb_map, coor_map, msk

    def rasterize_pyramid(self, coor_world, rgb, density=None, h=192, w=256, k=np.array([0.7, 1.2, 1.7, 2.2])):
        if density is None:
            density = torch.ones(coor_world.shape[0], 1).to(coor_world.device)
        mask = None
        image = None
        for ksize in k:
            img, _, msk = self.rasterize(coor_world, rgb, h, w, ksize, 10)
            mask = msk if mask is None else mask * msk
            image = img if image is None else image + img * mask.unsqueeze(-1).expand(img.shape)
        return image, mask


class VAE_encoder(torch_nn.Module):
    def __init__(self, data_dim, latent_dim, W=512, D=4):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.W = W
        self.D = D

        """Fully Connected Layers"""
        self.fc_layers = []
        current_dim = self.data_dim
        for i in range(self.D - 1):
            self.fc_layers.append(torch_nn.Linear(current_dim, self.W))
            current_dim = self.W
        self.fc_layers = torch_nn.ModuleList(self.fc_layers)

        self.fc_layer_mu = torch_nn.Linear(current_dim, self.latent_dim)
        self.fc_layer_log_var = torch_nn.Linear(current_dim, self.latent_dim)

    def forward(self, x):
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
        mu = self.fc_layer_mu(x)
        log_var = self.fc_layer_log_var(x)
        return mu, log_var


class VAE_decoder(torch_nn.Module):
    def __init__(self, data_dim, latent_dim, W=512, D=4):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.W = W
        self.D = D

        """Fully Connected Layers"""
        self.fc_layers = []
        current_dim = self.latent_dim
        for i in range(self.D - 1):
            self.fc_layers.append(torch_nn.Linear(current_dim, self.W))
            current_dim = self.W
        self.fc_layers = torch_nn.ModuleList(self.fc_layers)
        self.output_layer = torch_nn.Linear(current_dim, self.data_dim)

    def forward(self, x):
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


def reparameterize(mu, log_var, factor=1.):
    std = torch.exp(0.5 * log_var) * factor
    eps = torch.randn_like(std)
    return eps * std + mu


def reparameterize_jt(mu, log_var, factor=1.):
    std = jt.exp(0.5 * log_var) * factor
    eps = jt.randn_like(std)
    return eps * std + mu


class VAE(torch_nn.Module):
    def __init__(self, data_dim, latent_dim, W=512, D=4, kl_lambda=0.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.W = W
        self.D = D
        self.kl_lambda = kl_lambda
        self.encoder = VAE_encoder(data_dim=data_dim, latent_dim=latent_dim, W=W, D=D)
        self.decoder = VAE_decoder(data_dim=data_dim, latent_dim=latent_dim, W=W, D=D)

    def forward(self, x, various=True):
        """Forward Function"""
        z, mu, log_var = self.encode(x, various)
        y = self.decode(z)
        return y, z, mu, log_var

    def recon(self, x, various=False):
        """Reconstruction shapes"""
        z, _, _ = self.encode(x, various)
        y = self.decode(z)
        return y

    def encode(self, x, various=True):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var) if various else mu
        return z, mu, log_var

    def decode(self, z):
        y = self.decoder(z)
        return y

    def loss(self, x, y, mu, log_var, return_losses=False):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        recon_loss = torch.sum(torch.mean(torch.square(x - y), dim=0))
        loss = recon_loss + self.kl_lambda * kl_loss
        if return_losses:
            return loss, recon_loss, self.kl_lambda * kl_loss
        else:
            return loss

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples


class StyleLatents_variational(Module):
    def __init__(self, **kwargs):
        super().__init__()
        style_num, frame_num, latent_dim = kwargs['style_num'], kwargs['frame_num'], kwargs['latent_dim']
        self.style_num = style_num
        self.frame_num = frame_num
        self.latent_dim = latent_dim
        self.latents = jt.Var(jt.randn(self.style_num, self.frame_num, self.latent_dim))
        self.style_latents_mu = jt.Var(jt.randn(self.style_num, self.latent_dim))
        self.style_latents_logvar = jt.Var(jt.randn(self.style_num, self.latent_dim))
        self.sigma_scale = 1.
        self.set_requires_grad()
        self.latent_optimizer = None

    def set_requires_grad(self):
        self.latents.requires_grad = True
        self.style_latents_mu.requires_grad = False
        self.style_latents_logvar.requires_grad = False

    def rescale_sigma(self, sigma_scale=1.):
        self.sigma_scale = sigma_scale

    def execute(self, **kwargs):
        # style_ids, frame_ids of shape [batch]
        style_ids, frame_ids = kwargs['style_ids'], kwargs['frame_ids']
        flat_ids = style_ids * self.frame_num + frame_ids  # [batch]
        latents = self.latents.reshape([-1, self.latent_dim])[flat_ids]  # [batch, latent_dim]
        mu = self.style_latents_mu[style_ids]
        latents = mu + self.sigma_scale * (latents - mu)
        return latents

    def minus_logp(self, **kwargs):
        epsilon = 1e-3
        style_ids, frame_ids = kwargs['style_ids'], kwargs['frame_ids']
        latents = self(style_ids=style_ids, frame_ids=frame_ids)
        mu = self.style_latents_mu[style_ids]
        logvar = self.style_latents_logvar[style_ids]
        loss_logp = jt.sum((latents - mu.detach()) ** 2 / (jt.exp(0.5 * logvar.detach()) + epsilon), -1).mean()
        return loss_logp

    def set_latents(self):
        all_style_latents_mu, all_style_latents_logvar = self.style_latents_mu.unsqueeze(1).expand(list(self.latents.shape)),\
                                                         self.style_latents_logvar.unsqueeze(1).expand(list(self.latents.shape))
        latents = reparameterize_jt(all_style_latents_mu, all_style_latents_logvar, factor=1.)
        self.latents = jt.Var(latents)
        self.set_requires_grad()

    def set_optimizer(self):
        self.latent_optimizer = jt.nn.Adam([self.latents], lr=1e-3)

    def optimize(self, loss):
        if self.latent_optimizer is not None:
            self.latent_optimizer.step(loss)
