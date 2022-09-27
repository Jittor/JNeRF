'''
no check
'''
import jittor as jt
from jittor import Function
from .c_class import *
from .global_header import proj_path
# from opt.svox2.svox2 import Rays, SparseGrid
jt.flags.use_cuda = 1


class sample_grid(Function):
    def __init__(self):
        super().__init__()

    def execute(self, data_density: jt.Var,
                data_sh: jt.Var,
                grid:SparseGridSpec,
                points: jt.Var,
                want_colors: bool,
                grid_coords:bool=False):
        '''

        '''
        if grid_coords:
            self._grid_offset = jt.zeros_like(grid._offset)
            self._grid_scaling = jt.ones_like(grid._scaling)
        else:
            gsz = jt.array(np.array(grid._links.shape[0]))
            self._grid_offset = grid._offset*gsz-0.5
            self._grid_scaling = grid._scaling*gsz

        out_density = jt.empty([points.shape[0], grid.density_data.shape[1]], points.dtype)
        out_sh = jt.empty([points.shape[0] if want_colors else 0, grid.sh_data.shape[1]], points.dtype)
        if grid._background_links is None:
            grid._background_links = jt.empty([0, 1])
        out_density, out_sh = jt.code(inputs=[grid.density_data, grid.sh_data, grid._links, self._grid_offset, self._grid_scaling, grid._background_links, grid.background_data, grid.basis_data, points,
                                              ], outputs=[out_density, out_sh], cuda_header='#include "sample_kernel.h"', cuda_src=f"""
        @alias(grid_density_data,in0)
        @alias(grid_sh_data,in1)
        @alias(grid_links,in2)
        @alias(grid_offset,in3)
        @alias(grid_scaling,in4)
        @alias(grid_background_links,in5)
        @alias(grid_background_data,in6)
        @alias(grid_basis_data,in7)
        @alias(points,in8)
        @alias(out_density,out0)
        @alias(out_sh,out1)
        bool want_colors ={'true' if want_colors else 'false'};

    

        const auto Q = points_shape0 * grid_sh_data_shape1;
        //LOGir<<CUDA_MAX_THREADS;
        const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        const int blocks_density = CUDA_N_BLOCKS_NEEDED(points_shape0, cuda_n_threads);
        sample_grid_density_kernel<<<blocks_density, cuda_n_threads, 0>>>(
            {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
            PackedVar32<float,2>(points),
            // Output
            PackedVar32<float,2>(out_density));
        if (want_colors) {{
        sample_grid_sh_kernel<<<blocks, cuda_n_threads, 0>>>(
            {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
            PackedVar32<float,2>(points),
            // Output
            PackedVar32<float,2>(out_sh));
        }}
        CUDA_CHECK_ERRORS;
        """)
        out_density.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        self._points = points.detach()
        self.grid = grid
        self.want_colors = want_colors
        self.density_need_grad = data_density.requires_grad
        self.sh_need_grad = data_sh.requires_grad
        return out_density, out_sh

    def grad(self, grad_out_density, grad_out_sh):
        points = self._points
        grid = self.grid
        grad_density_grid = jt.zeros_like(self.grid.density_data)
        grad_sh_grid = jt.zeros_like(self.grid.sh_data)
        breakpoint()
        grad_density_grid, grad_sh_grid = jt.code(inputs=[
            grid.density_data, grid.sh_data, grid._links, self._grid_offset, self._grid_scaling, grid._background_links, grid.background_data, grid.basis_data, points, grad_out_density, grad_out_sh
        ], outputs=[grad_density_grid, grad_sh_grid, ], cuda_header='#include "sample_kernel.h"', cuda_src=f"""
        @alias(grid_density_data,in0)
        @alias(grid_sh_data,in1)
        @alias(grid_links,in2)
        @alias(grid_offset,in3)
        @alias(grid_scaling,in4)
        @alias(grid_background_links,in5)
        @alias(grid_background_data,in6)
        @alias(grid_basis_data,in7)
        @alias(points,in8)
        @alias(grad_out_density,in9)
        @alias(grad_out_sh,in10)

        @alias(grad_density_out,out0)
        @alias(grad_sh_out,out1)
        bool want_colors ={'true' if self.want_colors else 'false'}; 
        const auto Q = points_shape0 * grid_sh_data_shape1;

        const int cuda_n_threads = std::min<int>(Q, CUDA_MAX_THREADS);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        const int blocks_density = CUDA_N_BLOCKS_NEEDED(points_shape0, cuda_n_threads);
       
        sample_grid_density_backward_kernel<<<blocks_density, cuda_n_threads, 0>>>(
           {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
            PackedVar32<float,2>(points),
            PackedVar32<float,2>(grad_out_density),
            //Output
            PackedVar32<float,2>(grad_density_out)
            );

        if (want_colors) {{
            sample_grid_sh_backward_kernel<<<blocks, cuda_n_threads, 0>>>(
            {{grid_density_data,grid_sh_data,grid_links,grid_offset,grid_scaling,grid_background_links,grid_background_data,{grid.basis_dim},{grid.basis_type},grid_basis_data}},
            PackedVar32<float,2>(points),
            PackedVar32<float,2>(grad_out_sh),
            //Output
            PackedVar64<float,2>(grad_sh_out));
        }}
        CUDA_CHECK_ERRORS;
        """)
        grad_density_grid.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        if not self.density_need_grad:
            grad_density_grid = None
        if not self.sh_need_grad:
            grad_sh_grid = None
        return grad_density_grid, grad_sh_grid, None, None, None


