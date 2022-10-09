import jittor as jt
import numpy as np
import os
import jittor as jt
from jittor import Function, exp, log
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class splat_grid_samples_nerf_max_nearest_neighbor(Function):
    def __init__(self, density_grad_header, padded_output_width=16, using_fp16=False):
        self.density_grad_header = density_grad_header
        self.n_density_grid_samples = 0
        self.padded_output_width = padded_output_width
        self.grad_type='float32'
        if using_fp16:
            self.grad_type='float16'

    def execute(self,  density_grid_indices, mlp_out, density_grid_tmp, n_density_grid_samples):
        self.n_density_grid_samples = n_density_grid_samples
        assert(self.grad_type==mlp_out.dtype)
        output, = jt.code([density_grid_indices, mlp_out], [density_grid_tmp], cuda_header=global_headers+self.density_grad_header+'#include"splat_grid_samples_nerf_max_nearest_neighbor.h"', cuda_src=f"""
        #define grad_t in1_type
        cudaStream_t stream=0;
        uint32_t n_density_grid_samples={self.n_density_grid_samples};
        uint32_t*density_grid_indices=(uint32_t*)in0_p;
        uint32_t padded_output_width={self.padded_output_width};
        grad_t* mlp_out=(grad_t*)in1_p;
        float* density_grid_tmp=(float*)out0_p;

        ENerfActivation rgb_activation=ENerfActivation::Logistic;
        ENerfActivation density_activation=ENerfActivation::Exponential;
             linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor<grad_t>,0,stream,
            n_density_grid_samples, density_grid_indices, padded_output_width,mlp_out, density_grid_tmp,rgb_activation, density_activation);
        
        """)
        output.compile_options = proj_options
        output.sync()

        return output
