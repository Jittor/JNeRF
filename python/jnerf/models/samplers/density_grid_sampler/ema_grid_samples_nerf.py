import jittor as jt
import numpy as np

import os
import jittor as jt
from jittor import Function, exp, log
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class ema_grid_samples_nerf(Function):
    def __init__(self, density_grad_header, n_element=0, decay=0.95) -> None:
        self.density_grad_header = density_grad_header
        self.n_elements = n_element
        self.decay = decay

    def execute(self, density_grid_tmp, density_grid, n_elements):
        self.n_elements = n_elements
        output, = jt.code(inputs=[density_grid_tmp], outputs=[density_grid], 
        cuda_header=global_headers+self.density_grad_header+'#include "ema_grid_samples_nerf.h"', cuda_src=f"""
        uint32_t n_elements={self.n_elements};
        float * density_grid_tmp=(float*)in0_p;
        float *density_grid=(float*) out0_p;
        cudaStream_t stream=0;
        linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, {self.decay}, density_grid, density_grid_tmp);   
        """)
        output.compile_options = proj_options
        output.sync()
        return output
