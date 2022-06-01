
import jittor as jt
import numpy as np

import os
import jittor as jt
from jittor import Function
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class mark_untrained_density_grid(Function):
    def __init__(self, density_grad_header, n_images, image_resolutions) -> None:
        self.density_grad_header = density_grad_header
        self.n_elements =0
        self.n_images=n_images
        self.image_resolutions=image_resolutions

    def execute(self, focal_lengths,transforms,n_elements):
        self.n_elements=n_elements
        output=jt.zeros([self.n_elements],'float32')
        output,= jt.code( [focal_lengths,transforms],[output], cuda_header=global_headers+self.density_grad_header+'#include"mark_untrained_density_grid.h"', cuda_src=f"""
        uint32_t n_elements={self.n_elements};
        Eigen::Vector2i image_resolution{{{self.image_resolutions[0]},{self.image_resolutions[1]}}};
        cudaStream_t stream=0;
        float*density_grid=(float*)out0_p;
        int n_images={self.n_images};
        Eigen::Vector2f* focal_lengths_gpu=(Eigen::Vector2f*)in0_p;
        Eigen::Matrix<float, 3, 4>* transforms_gpu=(Eigen::Matrix<float, 3, 4>* )in1_p;
        linear_kernel(mark_untrained_density_grid, 0, stream, n_elements,density_grid,
						n_images,
                        focal_lengths_gpu,
			            transforms_gpu,
                        image_resolution);
        """)
        output.compile_options = proj_options
        output.sync()
        return output
