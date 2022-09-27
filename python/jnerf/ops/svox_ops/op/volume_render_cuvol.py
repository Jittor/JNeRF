'''

'''
import jittor as jt
from jittor import Function
from .c_class import *
from .global_header import proj_path
# from opt.svox2.svox2 import Rays, SparseGrid
jt.flags.use_cuda = 1


class volume_render_cuvol(Function):
    def __init__(self):
        super().__init__()
        self.color_cache = None
        self.grid = None
        self.rays = None
        self.opt = None
        self.basis_data = None
        self.sparse_grad_indexer=None
        pass

    def execute(self, data_density, data_sh, data_basis, data_background, grid, rays: RaysSpec, opt: RenderOptions,grid_coords:bool = False):
        '''

        '''


        if grid_coords:
            self._grid_offset = jt.zeros_like(grid._offset)
            self._grid_scaling = jt.ones_like(grid._scaling)
        else:
            gsz = jt.array(np.array(grid._links.shape[0]))
            self._grid_offset = grid._offset*gsz-0.5
            self._grid_scaling = grid._scaling*gsz

        if grid._background_links is None:
            grid._background_links = jt.empty([0, 1])
        Q: int = rays.origins.shape[0]
        results = jt.empty(rays.origins.shape)

        use_background: bool = (
            grid._background_links is not None) and grid._background_links.shape[0] > 0

        if grid._background_links is None:
            grid._background_links = jt.empty([0, 1])
        log_transmit: jt.Var = jt.empty([0, 1])
        if use_background:
            log_transmit = jt.empty(
                [rays.origins.shape[0]], dtype=rays.origins.dtype).stop_grad()

    
        results, = jt.code(inputs=[
            rays.origins, rays.dirs,  # in1
            grid.density_data, grid.sh_data, grid._links, self._grid_offset, self._grid_scaling, grid._background_links, grid.background_data, grid.basis_data  # in9
            , log_transmit
        ], outputs=[results], cuda_header='#include "volume_render_cuvol_fused.h"', cuda_src=f"""
        @alias(rays_origins,in0)
        @alias(rays_dirs,in1)
        @alias(grid_density_data,in2)
        @alias(grid_sh_data,in3)
        @alias(grid_links,in4)
        @alias(grid_offset,in5)
        @alias(grid_scaling,in6)
        @alias(grid_background_links,in7)
        @alias(grid_background_data,in8)
        @alias(grid_basis_data,in9)
        @alias(log_transmit,in10)
        @alias(output,out0)

        RenderOptions opt{{{opt.background_brightness},{opt.step_size},{opt.sigma_thresh},{opt.stop_thresh},{opt.near_clip},{1 if opt.use_spheric_clip else 0},{1 if opt.last_sample_opaque else 0}}};
    


        int Q = rays_origins_shape0;
        bool use_background = grid_background_links_shape0 > 0;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
       render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
                {{rays_origins,rays_dirs}}//rays
                ,opt
                ,PackedVar32<float,2>(output),
                use_background?log_transmit->ptr<float>():nullptr
                );
        if(use_background){{
            const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
            render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
                {{rays_origins,rays_dirs}},//rays
                opt,
                log_transmit_p,
                PackedVar32<float,2>(output));
        }}



        CUDA_CHECK_ERRORS;
        """)
        results.compile_options = {
            f"FLAGS: -I{proj_path}": 1}

        self.color_cache = results.detach()
        self.grid = grid
        self.rays = rays
        self.opt = opt
        self.basis_data = data_basis
        self.basis_data_grad = False
        self.background_data_grad = False
        if self.basis_data is not None and self.basis_data.shape[0] > 0:
            self.basis_data_grad = self.basis_data.requires_grad
        if self.grid.background_data is not None and self.grid.background_data.shape[0] > 0:
            self.background_data_grad = self.grid.background_data.requires_grad

        return results
        pass

    def grad(self, grad_out):

        opt = self.opt
        rays = self.rays
        grid = self.grid
        color_cache = self.color_cache
        grad_density_grid = jt.zeros_like(self.grid.density_data)
        grad_sh_grid = jt.zeros_like(self.grid.sh_data)

        if self.grid.basis_type == BASIS_TYPE_MLP:
            grad_basis = jt.zeros_like(self.basis_data)
        elif self.grid.basis_type == BASIS_TYPE_3D_TEXTURE:
            grad_basis = jt.zeros_like(self.grid.basis_data)
        if self.grid.background_data is not None and self.grid.background_data.shape[0] > 0:
            grad_background = jt.zeros_like(self.grid.background_data)
        
        
        if not self.basis_data_grad:
            grad_basis = jt.empty([0, 1])
        if not self.background_data_grad:
            grad_background = jt.empty([0, 1])

        # cpp
        Q: int = rays.origins.shape[0]
        # 原版c中torch接口
        use_background: bool = (
            grid._background_links is not None) and grid._background_links.shape[0] > 0
        log_transmit = jt.empty([0, 1])
        accum = jt.empty([0, 1])
        if use_background:
            log_transmit = jt.empty(
                [rays.origins.shape[0]], dtype=rays.origins.dtype).stop_grad()
            accum = jt.empty(
                [rays.origins.shape[0]], dtype=rays.origins.dtype).stop_grad()
        grad_density_grid, grad_sh_grid, grad_basis, grad_background = jt.code(inputs=[
            rays.origins, rays.dirs,  # in1
            grid.density_data, grid.sh_data, grid._links, self._grid_offset,self._grid_scaling, grid._background_links, grid.background_data, grid.basis_data  # in9
            , grad_out, log_transmit, accum, color_cache
        ], outputs=[grad_density_grid, grad_sh_grid, grad_basis, grad_background], cuda_header='#include "volume_render_cuvol_fused.h"', cuda_src=f"""
        @alias(rays_origins,in0)
        @alias(rays_dirs,in1)
        @alias(grid_density_data,in2)
        @alias(grid_sh_data,in3)
        @alias(grid_links,in4)
        @alias(grid_offset,in5)
        @alias(grid_scaling,in6)
        @alias(grid_background_links,in7)
        @alias(grid_background_data,in8)
        @alias(grid_basis_data,in9)
        @alias(grad_out,in10)
        @alias(log_transmit,in11)
        @alias(accum,in12)
        @alias(color_cache,in13)
        @alias(grad_density_grid,out0)
        @alias(grad_sh_grid,out1)
        @alias(grad_basis,out2)
        @alias(grad_background,out3)

        RenderOptions opt{{{opt.background_brightness},{opt.step_size},{opt.sigma_thresh},{opt.stop_thresh},{opt.near_clip},{1 if opt.use_spheric_clip else 0},{1 if opt.last_sample_opaque else 0}}};
        


        int Q = rays_origins_shape0;
        bool use_background = grid_background_links_shape0 > 0;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    PackedGridOutputGrads grads(grad_density_grid, grad_sh_grid, grad_basis, grad_background);

       render_ray_backward_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
                grad_out_p,
                color_cache_p,
                {{rays_origins,rays_dirs}}//rays
                ,opt,
                false,//TODO:fused
                nullptr,
                0.f,
                0.f,
                grads,
                use_background?accum->ptr<float>():nullptr,
                use_background?log_transmit->ptr<float>():nullptr
                );
        if(use_background){{
           const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
       render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
                   {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
                grad_out->ptr<float>(),
                color_cache->ptr<float>(),
               {{rays_origins,rays_dirs}},//rays
                opt,
                log_transmit->ptr<float>(),
                accum->ptr<float>(),
                false,
                0.f,
                // Output
                grads);
        }}



       //LOGir<<Q<<" "<<output->num<<" "<<output->size<<" "<<output->dsize()<<" "<<output->shape[0];
       //LOGir<<grid_background_links->num;
        CUDA_CHECK_ERRORS;//TODO:dy
        """)
        grad_density_grid.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        if not self.basis_data_grad:
            grad_basis = None
        if not self.background_data_grad:
            grad_background = None
        self.grid = self.rays = self.opt = None
        self.basis_data = None
        return grad_density_grid, grad_sh_grid, grad_basis, grad_background, None, None, None

