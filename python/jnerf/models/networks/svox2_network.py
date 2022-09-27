import jittor as jt
import jittor.nn as nn
from typing import Union,List,Tuple,Optional
from functools import reduce
import jnerf.utils.svox2_utils as utils
from jnerf.utils.svox2_utils import Rays,RenderOptions,get_expon_lr_func,Camera
from jnerf.ops.svox_ops.op.volume_render_cuvol import volume_render_cuvol
from jnerf.ops.svox_ops.op.tv_grad_sparse import tv_grad_sparse
from jnerf.ops.svox_ops.op.dilate import dilate
from jnerf.ops.svox_ops.op.grid_weight_render import grid_weight_render
from jnerf.ops.svox_ops.op.sample_grid import sample_grid
from tqdm import tqdm
BASIS_TYPE_SH = 1
import numpy as np

from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
@NETWORKS.register_module()
class SparseGrid(nn.Module):

    def __init__(
        self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        basis_type: int = BASIS_TYPE_SH,
        basis_dim: int = 9,  # SH/learned basis size; in SH case, square number
        basis_reso: int = 16,  # Learned basis resolution (x^3 embedding grid)
        use_z_order : bool=False,
        use_sphere_bound : bool=False,
        mlp_posenc_size : int = 0,
        mlp_width : int = 16,
        background_nlayers : int = 0,  # BG MSI layers
        background_reso : int = 256,  # BG MSI cubemap face size

    ):
        super().__init__()
        self.basis_type = basis_type
        assert basis_type == BASIS_TYPE_SH,"now only support sh basis type"
        if basis_type == BASIS_TYPE_SH:
            assert utils.isqrt(basis_dim) is not None, "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS
        ), f"basis_dim 1-{utils.MAX_SH_BASIS} supported"
        self.basis_dim = basis_dim

        self.mlp_posenc_size = mlp_posenc_size
        self.mlp_width = mlp_width

        self.background_nlayers = background_nlayers
        assert background_nlayers == 0 or background_nlayers > 1, "Please use at least 2 MSI layers (trilerp limitation)"
        self.background_reso = background_reso
        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert (
                len(reso) == 3
            ), "reso must be an integer or indexable object of 3 ints"

        if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
            print("Morton code requires a cube grid of power-of-2 size, ignoring...")
            use_z_order = False

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if not isinstance(radius, jt.Var):
            radius = jt.array(radius,dtype=jt.float32)
        if not isinstance(center, jt.Var):
            center = jt.array(center,dtype=jt.float32)

        self._radius: jt.Var = radius.stop_grad()  # CPU TODO:
        self._center: jt.Var = center.stop_grad()  # CPU
        self._offset = (0.5 * (1.0 - self._center / self._radius)).stop_grad()
        self._scaling = (0.5 / self._radius).stop_grad()
        n3: int = reduce(lambda x, y: x * y, reso)
        if use_z_order:
            init_links = utils.gen_morton(reso[0], dtype=jt.int32).flatten()
        else:
            init_links = jt.arange(n3, dtype=jt.int32)
        if use_sphere_bound:
            X = jt.arange(reso[0], dtype=jt.float32) - 0.5
            Y = jt.arange(reso[1], dtype=jt.float32) - 0.5
            Z = jt.arange(reso[2], dtype=jt.float32) - 0.5
            X, Y, Z = jt.meshgrid(X, Y, Z)
            points = jt.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = jt.array(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            points = roffset + points * rscaling

            norms = points.norm(dim=-1)
            mask = norms <= 1.0 + (3 ** 0.5) / gsz.max()
            self.capacity: int = mask.sum().item()
            # breakpoint()
            data_mask = jt.zeros(n3, dtype=jt.int32)
            idxs = init_links[mask].int64()
            data_mask[idxs] = 1
            data_mask = jt.cumsum(data_mask, dim=0) - 1

            init_links[mask] = data_mask[idxs].int()
            init_links[jt.logical_not(mask)] = -1
        else:
            self.capacity = n3

        self.density_data = nn.Parameter(
            jt.zeros([self.capacity, 1], dtype=jt.float32)
        )
        # Called sh for legacy reasons, but it's just the coeffients for whatever
        # spherical basis functions
        self.sh_data = nn.Parameter(
            jt.zeros(
                [self.capacity, self.basis_dim * 3], dtype=jt.float32
            )
        )

        if self.basis_type == BASIS_TYPE_SH:
            self.basis_data = nn.Parameter(
                jt.empty(
                    [0, 0, 0, 0], dtype=jt.float32
                ),
                requires_grad=False
            )
        self._background_links:Optional[jt.Var]
        self.background_data: Optional[jt.Var]
        if self.use_background:
            background_capacity = (self.background_reso ** 2) * 2
            background_links = jt.arange(
                background_capacity,
                dtype=jt.int32
            ).reshape(self.background_reso * 2, self.background_reso)
            self._background_links=background_links
            self.background_data = nn.Parameter(
                jt.zeros(
                    [background_capacity,
                    self.background_nlayers,
                    4],
                    dtype=jt.float32
                )
            )
        else:
            self._background_links=None
            self.background_data = nn.Parameter(
                jt.empty(
                    [0, 0, 0],
                    dtype=jt.float32
                ),
                requires_grad=False
            )
        self._links=init_links.view(reso).stop_grad()
        self.opt = utils.RenderOptions()
        self.sparse_grad_indexer: Optional[jt.Var] = None
        self.sparse_sh_grad_indexer: Optional[jt.Var] = None
        self.sparse_background_indexer: Optional[jt.Var] = None
        self.density_rms: Optional[jt.Var] = None
        self.sh_rms: Optional[jt.Var] = None
        self.background_rms: Optional[jt.Var] = None
        self.basis_rms: Optional[jt.Var] = None
        self.volume_render_cuvol=volume_render_cuvol()
        self.tv_grad_sparse=tv_grad_sparse()
        self.tv_grad_color_sparse=tv_grad_sparse()
        self.dilate=dilate()
        self.grid_weight_render=grid_weight_render()
        self.sample_grid = sample_grid()


    @property
    def use_background(self):
        return self.background_nlayers > 0

    def param_init(self,args):
        self.sh_data.data[:]=0
        self.density_data.data[:] = 0  if args.lr_fg_begin_step > 0 else args.init_sigma
        if self.use_background:
            self.background_data.data[..., -1] = args.init_sigma_bg

    def setup_render_opts(self, args):
        """
        Pass render arguments to the SparseGrid renderer options
        """
        self.opt.step_size = args.step_size
        self.opt.sigma_thresh = args.sigma_thresh
        self.opt.stop_thresh = args.stop_thresh
        self.opt.background_brightness = args.background_brightness
        self.opt.backend = args.renderer_backend
        self.opt.random_sigma_std = args.random_sigma_std
        self.opt.random_sigma_std_background = args.random_sigma_std_background
        self.opt.last_sample_opaque = args.last_sample_opaque
        self.opt.near_clip = args.near_clip
        self.opt.use_spheric_clip = args.use_spheric_clip
    def volume_render_jt(self,
        rays: Rays,
        randomize: bool = False):
        basis_data=jt.empty([0,1])
        if self.opt.backend =='cuvol':
            rgb_out=self.volume_render_cuvol(
                self.density_data,
                self.sh_data,
                basis_data,
                self.background_data if self.use_background else None,
                self,
                rays,
                self.opt
            )
        return rgb_out
    def volume_render_image(
        self, camera: Camera, use_kernel: bool = True, randomize: bool = False,
        batch_size : int = 5000,
        return_raylen: bool=False
    ):
        """
        Standard volume rendering (entire image version).
        See grid.opt.* (RenderOptions) for configs.

        :param camera: Camera
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :return: (H, W, 3), predicted RGB image
        """
      

        # Manually generate rays for now
        rays = camera.gen_rays()
        all_rgb_out = []
        for batch_start in range(0, camera.height * camera.width, batch_size):
            rgb_out_part = self.volume_render_jt(rays[batch_start:batch_start+batch_size],
                                                
                                                randomize=randomize,
                                                )
            all_rgb_out.append(rgb_out_part)
        all_rgb_out = jt.concat(all_rgb_out, dim=0)
        return all_rgb_out.view(camera.height, camera.width, -1)

    def inplace_tv_grad(self, grad: jt.Var,
                        scaling: float = 1.0,
                        sparse_frac: float = 0.01,
                        logalpha: bool=False, logalpha_delta: float=2.0,
                        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
                        contiguous: bool = True
                    ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert not logalpha, "No longer supported"
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                grad,mask_out= self.tv_grad_sparse.execute(self._links,self.density_data,rand_cells,
                self._get_sparse_grad_indexer(),0,1,scaling,False,self.opt.last_sample_opaque,grad)
                return grad
          
       

    def inplace_tv_color_grad(
        self,
        grad: jt.Var,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool=False,
        logalpha_delta: float=2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.
        """
      
        assert not logalpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_sh_grad_indexer()
              
                grad,mask_out=self.tv_grad_color_sparse.execute(self._links,self.sh_data,
                rand_cells,indexer,start_dim,end_dim,scaling,True,False,grad)
                return grad
              
    def _get_rand_cells(self, sparse_frac: float, force: bool = False, contiguous:bool=True):
        if sparse_frac < 1.0 or force:
            assert self.sparse_grad_indexer is None or self.sparse_grad_indexer.dtype == jt.bool, \
                   "please call sparse loss after rendering and before gradient updates"
            grid_size = self._links.size(0) * self._links.size(1) * self._links.size(2)
            sparse_num = max(int(sparse_frac * grid_size), 1)
            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = jt.arange(start, start + sparse_num, dtype=jt.int32)

                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start:] -= grid_size
                return arr
            else:
                return jt.randint(0, grid_size, (sparse_num,), dtype=jt.int32)
        return None

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = jt.empty((0,), dtype=jt.bool)
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = jt.empty((0,), dtype=jt.bool)
        return indexer

    def resample(
        self,
        reso: Union[int, List[int]],
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[Camera]] = None,
        use_z_order: bool=False,
        accelerate: bool=True,
        weight_render_stop_thresh: float = 0.2, # SHOOT, forgot to turn this off for main exps..
        max_elements:int=0
    ):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
                             (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
                           to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
                                                 0.0 = no thresholding, 1.0 = hides everything.
                                                 Useful for force-cutting off
                                                 junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
                upsampled grid; we will adjust the threshold to match it
        """
        print("start resample")
        with jt.no_grad():
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
                print("Morton code requires a cube grid of power-of-2 size, ignoring...")
                use_z_order = False

            self.capacity: int = reduce(lambda x, y: x * y, reso)
            curr_reso = self._links.shape
            dtype = jt.float32
            reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]
            X = jt.linspace(
                reso_facts[0] - 0.5,
                curr_reso[0] - reso_facts[0] - 0.5,
                reso[0],       
            ).astype(dtype)
            Y = jt.linspace(
                reso_facts[1] - 0.5,
                curr_reso[1] - reso_facts[1] - 0.5,
                reso[1],
     
            ).astype(dtype)
            Z = jt.linspace(
                reso_facts[2] - 0.5,
                curr_reso[2] - reso_facts[2] - 0.5,
                reso[2],
          
            ).astype(dtype)
            X, Y, Z = jt.meshgrid(X, Y, Z)
            points = jt.stack((X, Y, Z), dim=-1).view(-1, 3)

            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=jt.int64).view(-1)
                points[morton] = points.clone()
            use_weight_thresh = cameras is not None

            batch_size = 720720
            all_sample_vals_density = []
            print('Pass 1/2 (density)')
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density, _ = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=False
                )
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)

            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None

            sample_vals_density = jt.concat(
                    all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = jt.array(reso)
                offset = self._offset * gsz - 0.5
                scaling = self._scaling * gsz
                max_wt_grid = jt.zeros(reso, dtype=jt.float32)
                print(" Grid weight render", sample_vals_density.shape)            
                for i, cam in tqdm(enumerate(cameras)):    
                    max_wt_grid=self.grid_weight_render.execute(sample_vals_density,cam._to_op(),0.5,weight_render_stop_thresh,False,offset,scaling,max_wt_grid)
                    jt.sync_all(True)
                sample_vals_mask = max_wt_grid >= weight_thresh
                if max_elements > 0 and max_elements < max_wt_grid.numel() \
                                    and max_elements < (sample_vals_mask!=0).sum():
                    # To bound the memory usage
                    weight_thresh_bounded = jt.topk(max_wt_grid.view(-1),
                                     k=max_elements, sorted=False)[0].min().item()
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    print(' Readjusted weight thresh to fit to memory:', weight_thresh)
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if max_elements > 0 and max_elements < sample_vals_density.numel() \
                                    and max_elements < (sample_vals_mask!=0).sum():
                    # To bound the memory usage
                    sigma_thresh_bounded = jt.topk(sample_vals_density.view(-1),
                                     k=max_elements, sorted=False)[0].min().item()
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    print(' Readjusted sigma thresh to fit to memory:', sigma_thresh)
                    sample_vals_mask = sample_vals_density >= sigma_thresh

                if self.opt.last_sample_opaque:
                    # Don't delete the last z layer
                    sample_vals_mask[:, :, -1] = 1

            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = self.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = (sample_vals_mask!=0).sum().item()

            # Now we can get the colors for the sparse points
            points = points[sample_vals_mask]
            print('Pass 2/2 (color), eval', cnz, 'sparse pts')
            all_sample_vals_sh = []
            for i in tqdm(range(0, len(points), batch_size)):
                _, sample_vals_sh = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=True
                )
                all_sample_vals_sh.append(sample_vals_sh)
            sample_vals_sh =jt.concat(all_sample_vals_sh, dim=0) if len(all_sample_vals_sh) else jt.empty_like(self.sh_data[:0])
            del self.density_data
            del self.sh_data
            del all_sample_vals_sh

            if use_z_order:
                inv_morton = jt.empty(morton.shape).astype(morton.dtype)
                inv_morton[morton] = jt.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = jt.full(
                    (sample_vals_mask.size(0),), fill_value=-1, dtype=jt.int32
                )
                init_links[inv_idx] = jt.arange(inv_idx.size(0), dtype=jt.int32)
            else:
                init_links = (
                    jt.cumsum(sample_vals_mask.astype(jt.int32), dim=-1).int() - 1
                )
                init_links[jt.logical_not(sample_vals_mask)] = -1

            self.capacity = cnz
            print(" New cap:", self.capacity)
            del sample_vals_mask
            print('density', sample_vals_density.shape, sample_vals_density.dtype)
            print('sh', sample_vals_sh.shape, sample_vals_sh.dtype)
            print('links', init_links.shape, init_links.dtype)
            self.density_data = nn.Parameter(sample_vals_density.view(-1, 1))
            self.sh_data = nn.Parameter(sample_vals_sh)
            self._links = init_links.view(reso)

    
    def sample(self, points: jt.Var,
                use_kernel:bool = True,
               grid_coords: bool = False,
               want_colors: bool = True):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: jt.Var, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates ([-0.5, reso[i]-0.5 ] in each dimension);
                                  more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
                            else returns density and a dummy tensor to be ignored
                            (much faster)

        :return: (density, color)
        """
        if use_kernel:
            return self.sample_grid( self.density_data, self.sh_data, self, points, want_colors,grid_coords)

        else:
            if not grid_coords:
                points = self.world2grid(points)

            points.assign(jt.clamp(points,min_v=0.0))
            for i in range(3):
                points[:, i]= jt.clamp(points[:, i],max_v=self._links.size(i) - 1)
            l = points.astype(jt.int64)
            for i in range(3):
                l[:, i]= jt.clamp(l[:, i],max_v=self._links.size(i) - 2)
            wb = points - l
            wb = wb.astype(jt.float32)
            wa = 1.0 - wb

            lx, ly, lz = l.unbind(-1)
            links000 = self._links[lx, ly, lz]
            links001 = self._links[lx, ly, lz + 1]
            links010 = self._links[lx, ly + 1, lz]
            links011 = self._links[lx, ly + 1, lz + 1]
            links100 = self._links[lx + 1, ly, lz]
            links101 = self._links[lx + 1, ly, lz + 1]
            links110 = self._links[lx + 1, ly + 1, lz]
            links111 = self._links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = jt.empty_like(self.sh_data[:0])

            return samples_sigma, samples_rgb

    def execute(self, points: jt.Var, use_kernel: bool = True):
        return self.sample(points, use_kernel=use_kernel)
    
    def save(self, path: str, compress: bool = False):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        data = {
            "radius":self._radius.numpy(),
            "center":self._center.numpy(),
            "links":self._links.numpy(),
            "density_data":self.density_data.numpy(),
            "sh_data":self.sh_data.numpy().astype(np.float16),
        }
        data['basis_type'] = self.basis_type

        save_fn(
            path,
            **data
        )

    @classmethod
    def load(cls, path: str):
        """
        Load from path
        """
        z = np.load(path)
        if "data" in z.keys():
            # Compatibility
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data

        if 'background_data' in z:
            background_data = z['background_data']
            background_links = z['background_links']
        else:
            background_data = None

        links = z.f.links
        basis_dim = (sh_data.shape[1]) // 3
        radius = z.f.radius.tolist() if "radius" in z.files else [1.0, 1.0, 1.0]
        center = z.f.center.tolist() if "center" in z.files else [0.0, 0.0, 0.0]
        grid = cls(
            1,
            radius=radius,
            center=center,
            basis_dim=basis_dim,
            use_z_order=False,
            basis_type=z['basis_type'].item() if 'basis_type' in z else BASIS_TYPE_SH,
            mlp_posenc_size=z['mlp_posenc_size'].item() if 'mlp_posenc_size' in z else 0,
            mlp_width=z['mlp_width'].item() if 'mlp_width' in z else 16,
            background_nlayers=0,
        )
        if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
        if density_data.dtype != np.float32:
            density_data = density_data.astype(np.float32)
        sh_data = jt.array(sh_data)
        density_data = jt.array(density_data)
        grid.sh_data = nn.Parameter(sh_data)
        grid.density_data = nn.Parameter(density_data)
        grid._links = jt.array(links)
        grid.capacity = grid.sh_data.size(0)
        grid.basis_data = nn.Parameter(grid.basis_data)
        return grid