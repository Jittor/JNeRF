import jittor as jt
from jittor import Function
from jnerf.ops.code_ops.global_vars import global_headers, proj_options
jt.flags.use_cuda = 1

class generate_grid_samples_nerf_nonuniform(Function):
    def __init__(self, density_grad_header, aabb_range,max_cascade=0,density_grid_ema_step=0,n_element=0,) -> None:
        self.density_grad_header = density_grad_header
        self.n_elements = n_element
        self.density_grid_ema_step = density_grid_ema_step
        self.max_cascade = max_cascade
        self.aabb_range=aabb_range
        self.thresh=-0.01

    def execute(self, density_grid, n_elements, density_grid_ema_step, max_cascade, thresh) -> None:
        self.n_elements=n_elements
        self.density_grid_ema_step=density_grid_ema_step
        self.max_cascade=max_cascade
        self.thresh=thresh
        output= jt.code(shapes=[(self.n_elements,3), (self.n_elements,)],dtypes= [density_grid.dtype,jt.int32],inputs= [density_grid,density_grid_ema_step], 
        cuda_header=global_headers+self.density_grad_header+'#include "generate_grid_samples_nerf_nonuniform.h"', cuda_src=f"""
        @alias(density_grid_ema_step,in1)
        uint32_t n_elements=out0_shape0;
        uint32_t max_cascade={self.max_cascade};
        float* density_grid=(float*)in0_p;
        cudaStream_t stream=0;
        NerfPosition* density_grid_positions=(NerfPosition*)out0_p;

        
        uint32_t* density_grid_indices=(uint32_t*)out1_p;
        BoundingBox m_aabb = BoundingBox{{Vector3f::Constant({self.aabb_range[0]}), Vector3f::Constant({self.aabb_range[1]})}};
        float thresh={self.thresh};
        linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_elements,
			rng,
			(const uint32*)density_grid_ema_step_p,
			m_aabb,
			density_grid,
			density_grid_positions,
			density_grid_indices,
			max_cascade+1,
			thresh
		);
        rng.advance();


        
        """)
        output[0].compile_options = proj_options
        output[0].sync()
        output[1].sync()
        return output




