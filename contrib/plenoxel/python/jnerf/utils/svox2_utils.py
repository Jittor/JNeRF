from functools import partial
# import torch
# from torch import nn
import jittor as jt
from jittor import nn
from typing import List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import math

def inthroot(x : int, n : int):
    if x <= 0:
        return None
    lo, hi = 1, x
    while lo <= hi:
        mi = lo + (hi - lo) // 2
        p = mi ** n
        if p == x:
            return mi
        elif p > x:
            hi = mi - 1
        else:
            lo = mi + 1
    return None

isqrt = partial(inthroot, n=2)

def is_pow2(x : int):
    return x > 0 and (x & (x - 1)) == 0

# def _get_c_extension():
#     from warnings import warn
#     try:
#         import svox2.csrc as _C
#         if not hasattr(_C, "sample_grid"):
#             _C = None
#     except:
#         _C = None

#     if _C is None:
#         warn("CUDA extension svox2.csrc could not be loaded! " +
#              "Operations will be slow.\n" +
#              "Please do not import svox in the svox2 source directory.")
#     return _C


# Morton code (Z-order curve)
def _expand_bits(v):
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
    return v

def _unexpand_bits(v):
    v &= 0x49249249
    v = (v | (v >> 2)) & 0xc30c30c3
    v = (v | (v >> 4)) & 0xf00f00f
    v = (v | (v >> 8)) & 0xff0000ff
    v = (v | (v >> 16)) & 0x0000ffff
    return v


def morton_code_3(x, y, z):
    xx = _expand_bits(x)
    yy = _expand_bits(y)
    zz = _expand_bits(z)
    return (xx << 2) + (yy << 1) + zz

def inv_morton_code_3(code):
    x = _unexpand_bits(code >> 2)
    y = _unexpand_bits(code >> 1)
    z = _unexpand_bits(code)
    return x, y, z

def gen_morton(D, dtype=jt.int64):
    assert is_pow2(D), "Morton code requires power of 2 reso"
    arr = jt.arange(D, dtype=dtype)
    X, Y, Z = jt.meshgrid(arr, arr, arr)
    mort = morton_code_3(X, Y, Z)
    return mort


# SH

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10
def eval_sh_bases(basis_dim : int, dirs : jt.Var):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    #TODO:????
    result = jt.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result


def spher2cart(theta : jt.Var, phi : jt.Var):
    """Convert spherical coordinates into Cartesian coordinates on unit sphere."""
    x = jt.sin(theta) * jt.cos(phi)
    y = jt.sin(theta) * jt.sin(phi)
    z = jt.cos(theta)
    return jt.stack([x, y, z], dim=-1)

def eval_sg_at_dirs(sg_lambda : jt.Var, sg_mu : jt.Var, dirs : jt.Var):
    """
    Evaluate spherical Gaussian functions at unit directions
    using learnable SG basis,
    without taking linear combination
    Works with torch.
    ... Can be 0 or more batch dimensions.
    N is the number of SG basis we use.
    :math:`Output = \sigma_{i}{exp ^ {\lambda_i * (\dot(\mu_i, \dirs) - 1)}`

    :param sg_lambda: The sharpness of the SG lobes. (N), positive
    :param sg_mu: The directions of the SG lobes. (N, 3), unit vector
    :param dirs: jnp.ndarray unit directions (..., 3)

    :return: (..., N)
    """
    product = jt.linalg.einsum(
        "ij,...j->...i", sg_mu, dirs)  # [..., N]
    basis = jt.exp(jt.linalg.einsum(
        "i,...i->...i", sg_lambda, product - 1))  # [..., N]
    return basis

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.0)
        m.bias=jt.zeros_like(m.bias)


def cross_broadcast(x , y):
    """
    Cross broadcasting for 2 tensors

  
    :return: tuple of cross-broadcasted tensors x, y. Any dimension where the size of x or y is 1
             is expanded to the maximum size in that dimension among the 2.
             Formally, say the shape of x is (a1, ... an)
             and of y is (b1, ... bn);
             then the result has shape (a'1, ... a'n), (b'1, ... b'n)
             where
                :code:`a'i = ai if (ai > 1 and bi > 1) else max(ai, bi)`
                :code:`b'i = bi if (ai > 1 and bi > 1) else max(ai, bi)`
    """
    assert x.ndim == y.ndim, "Only available if ndim is same for all tensors"
    max_shape = [(-1 if (a > 1 and b > 1) else max(a,b)) for i, (a, b)
                    in enumerate(zip(x.shape, y.shape))]
    shape_x = [max(a, m) for m, a in zip(max_shape, x.shape)]
    shape_y = [max(b, m) for m, b in zip(max_shape, y.shape)]
    #TODO:
    x = x.broadcast_to(shape_x)
    y = y.broadcast_to(shape_y)
    return x, y

def posenc(
    x,
    cov_diag: Optional[jt.Var],
    min_deg: int,
    max_deg: int,
    include_identity: bool = True,
    enable_ipe: bool = True,
    cutoff: float = 1.0,
):
    """
    Positional encoding function. Adapted from jaxNeRF
    (https://github.com/google-research/google-research/tree/master/jaxnerf).
    With support for mip-NeFF IPE (by passing cov_diag != 0, keeping enable_ipe=True).
    And BARF-nerfies frequency attenuation (setting cutoff)

    Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1],
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    :param x: torch.Tensor (..., D), variables to be encoded. Note that x should be in [-pi, pi].
    :param cov_diag: torch.Tensor (..., D), diagonal cov for each variable to be encoded (IPE)
    :param min_deg: int, the minimum (inclusive) degree of the encoding.
    :param max_deg: int, the maximum (exclusive) degree of the encoding. if min_deg >= max_deg,
                         positional encoding is disabled.
    :param include_identity: bool, if true then concatenates the identity
    :param enable_ipe: bool, if true then uses cov_diag to compute IPE, if available.
                             Note cov_diag = 0 will give the same effect.
    :param cutoff: float, in [0, 1], a relative frequency cutoff as in BARF/nerfies. 1 = all frequencies,
                          0 = no frequencies

    :return: encoded torch.Tensor (..., D * (max_deg - min_deg) * 2 [+ D if include_identity]),
                     encoded variables.
    """
    if min_deg >= max_deg:
        return x
    scales = jt.array([2 ** i for i in range(min_deg, max_deg)])
    half_enc_dim = x.shape[-1] * scales.shape[0]
    shapeb = list(x.shape[:-1]) + [half_enc_dim]  # (..., D * (max_deg - min_deg))
    xb = jt.reshape((x[..., None, :] * scales[:, None]), shapeb)
    four_feat = jt.sin(
        jt.concat([xb, xb + 0.5 * np.pi], dim=-1)
    )  # (..., D * (max_deg - min_deg) * 2)
    if enable_ipe and cov_diag is not None:
        # Apply integrated positional encoding (IPE)
        xb_var = jt.reshape((cov_diag[..., None, :] * scales[:, None] ** 2), shapeb)
        xb_var = jt.repeat(xb_var, (2,))  # (..., D * (max_deg - min_deg) * 2)
        four_feat = four_feat * jt.exp(-0.5 * xb_var)
    if cutoff < 1.0:
        # BARF/nerfies, could be made cleaner
        cutoff_mask = _cutoff_mask( #TODO:????
            scales, cutoff * (max_deg - min_deg)
        )  # (max_deg - min_deg,)
        four_feat = four_feat.view(shapeb[:-1] + [2, scales.shape[0], x.shape[-1]])
        four_feat = four_feat * cutoff_mask[..., None]
        four_feat = four_feat.view(shapeb[:-1] + [2 * scales.shape[0] * x.shape[-1]])
    if include_identity:
        four_feat = jt.concat([x] + [four_feat], dim=-1)
    return four_feat


def net_to_dict(out_dict : dict,
                prefix : str,
                model : nn.Module):
    for child in model.named_children():
        layer_name = child[0]
        layer_params = {}
        for param in child[1].named_parameters():
            param_name = param[0]
            param_value = param[1].data.cpu().numpy()
            out_dict['pt__' + prefix + '__' + layer_name + '__' + param_name] = param_value




def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz

    origins = jt.stack([o0, o1, o2], -1)
    directions = jt.stack([d0, d1, d2], -1)
    return origins, directions


def xyz2equirect(bearings, reso):
    """
    Convert ray direction vectors into equirectangular pixel coordinates.
    Inverse of equirect2xyz.
    Taken from Vickie Ye
    """
    lat = jt.asin(bearings[..., 1])
    lon = jt.atan2(bearings[..., 0], bearings[..., 2])
    x = reso * 2 * (0.5 + lon / 2 / np.pi)
    y = reso * (0.5 - lat / np.pi)
    return jt.stack([x, y], dim=-1)

@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = "cuvol"  # One of cuvol, svox1, nvol

    background_brightness: float = 1.0  # [0, 1], the background color black-white

    step_size: float = 0.5  # Step size, in normalized voxels (not used for svox1)
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh: float = 1e-10  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    stop_thresh: float = (
        1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False   # Make the last sample opaque (for forward-facing)

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    random_sigma_std: float = 1.0                   # Noise to add to sigma (only if randomize=True)
    random_sigma_std_background: float = 1.0        # Noise to add to sigma
                                                    # (for the BG model; only if randomize=True)

@dataclass
class Rays:
    origins: jt.Var
    dirs: jt.Var

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

@dataclass
class Camera:
    c2w: jt.Var  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_op(self):
        spec = CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx_val
        spec.fy = self.fy_val
        spec.cx = self.cx_val
        spec.cy = self.cy_val
        spec.width = self.width
        spec.height = self.height
        spec.ndc_coeffx = self.ndc_coeffs[0]
        spec.ndc_coeffy = self.ndc_coeffs[1]
        return spec

    def gen_rays(self) -> Rays:
        """
        Generate the rays for this camera
        :return: (origins (H*W, 3), dirs (H*W, 3))
        """

        origins = self.c2w[None, :3, 3].expand(self.height * self.width, -1)
        yy, xx = jt.meshgrid(
            jt.arange(self.height, dtype=jt.float64) + 0.5,
            jt.arange(self.width, dtype=jt.float64) + 0.5,
        )
        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = jt.ones_like(xx)
        dirs = jt.stack((xx, yy, zz), dim=-1)   # OpenCV
        del xx, yy, zz
        dirs /= jt.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)

        dirs = (self.c2w[None, :3, :3].expand(dirs.shape[0],1,1) @ dirs.float32())[..., 0]

        dirs = dirs.reshape(-1, 3).float()

        if self.ndc_coeffs[0] > 0.0:
            origins, dirs = convert_to_ndc(
                    origins,
                    dirs,
                    self.ndc_coeffs)
            dirs /= jt.norm(dirs, dim=-1, keepdim=True)
        return Rays(origins, dirs)




## c_class.py
class CameraSpec():
    def __init__(self) -> None:
        self.c2w = None
        self.fx  = 0.
        self.fy = 0.
        self.cx = 0.
        self.cy = 0.
        self.width = 0
        self.height = 0
        self.ndc_coeffx = 0.
        self.ndc_coeffy = 0.

def select_or_shuffle_rays(rays_init: Rays,
                           permutation: int = False,
                           epoch_size: Optional[int] = None,
                           ):
    n_rays = rays_init.origins.size(0)
    n_samp = n_rays if (epoch_size is None) else epoch_size
    if permutation:
        print(" Shuffling rays")
        indexer = jt.randperm(n_rays)[:n_samp]
    else:
        print(" Selecting random rays")
        indexer = jt.randint(0,n_rays,shape= [n_samp,])
    # print(indexer.shape)
    # indexer=jt.arange(n_samp)%n_rays
    # print(indexer.shape)
    return rays_init[indexer]

@dataclass
class Intrin:
    fx: Union[float, jt.Var]
    fy: Union[float, jt.Var]
    cx: Union[float, jt.Var]
    cy: Union[float, jt.Var]

    def scale(self, scaling: float):
        return Intrin(
            self.fx * scaling,
            self.fy * scaling,
            self.cx * scaling,
            self.cy * scaling
        )

    def get(self, field: str, image_id: int = 0):
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()

@dataclass
class RaysDataset:
    origins: Union[jt.Var, List[jt.Var]]
    dirs: Union[jt.Var, List[jt.Var]]
    gt: Union[jt.Var, List[jt.Var]]

    def to(self, *args, **kwargs):
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        return RaysDataset(origins, dirs, gt)

    def __getitem__(self, key):
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        return RaysDataset(origins, dirs, gt)

    def __len__(self):
        return self.origins.size(0)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper