import jittor as jt
import numpy as np
import os
import jittor as jt
from jittor import Function, exp, log
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class update_bitfield(Function):
    def __init__(self, density_grad_header):
        self.density_grad_header = density_grad_header

    def execute(self, density_grid, density_grid_mean, density_grid_bitfield):
        density_grid_bitfield, density_grid_mean = jt.code([density_grid, density_grid_mean], [density_grid_bitfield, density_grid_mean], cuda_header=global_headers+self.density_grad_header+'#include"update_bitfield.h"', cuda_src=f"""
        cudaStream_t stream=0;
        const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();
        
	    size_t size_including_mips =grid_mip_offset(NERF_CASCADES())/8;
       
        float* density_grid=(float*) in0_p;
        float *density_grid_mean=(float*) out1_p;
        cudaMemsetAsync(out1_p, 0, out1->size);
        uint8_t* density_grid_bitfield=(uint8_t*)out0_p;
        reduce_sum(
		density_grid,[n_elements] __device__(float val)
		{{ return fmaxf(val, 0.f) / (n_elements); }},
		density_grid_mean, n_elements, stream);
	    linear_kernel(grid_to_bitfield, 0, stream, n_elements / 8 * NERF_CASCADES(), density_grid, density_grid_bitfield, density_grid_mean);
        for (uint32_t level = 1; level < NERF_CASCADES(); ++level)
	    {{
           
		linear_kernel(bitfield_max_pool, 0, stream, n_elements / 64, density_grid_bitfield +grid_mip_offset(level-1)/8, density_grid_bitfield + grid_mip_offset(level) / 8);
        
        }}
        """)
        density_grid_bitfield.compile_options = proj_options
        density_grid_bitfield.sync()

        return density_grid_bitfield,density_grid_mean
