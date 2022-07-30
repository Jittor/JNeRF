import os
import copy
import pathlib
import jittor as jt
import jnerf
from jittor import Function, exp, log
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class CalcRgb(Function):
    def __init__(self, density_grad_header, aabb_range=(-1.5, 2.5), n_rays_per_batch=4096, n_rays_step=1024, padded_output_width=4, bg_color=[1, 1, 1], using_fp16=False):
        self.density_grad_header = density_grad_header
        self.bg_color = bg_color
        self.aabb_range = aabb_range

        self.n_rays_per_batch = n_rays_per_batch
        self.padded_output_width = padded_output_width
        self.num_elements = n_rays_per_batch*n_rays_step
        # activation 0:None 1:relu 2:sigmoid 3:exp
        self.rgb_activation = 2
        self.density_activation = 3
        self.ray_numstep_counter = jt.zeros([2], 'int32')
        self.code_path = pathlib.Path(__file__).parent.resolve()
        self.rgb_options = copy.deepcopy(proj_options)
        if using_fp16:
            self.grad_type = 'float16'
        else:
            self.grad_type = 'float32'

    def execute(self, network_output, coords_in, rays_numsteps, density_grid_mean, rays_numsteps_compacted,training_background_color):
        # input
        # network_output num_elements x 4 fp16 maybe
        # coords_in n_rays_per_batch x 7
        # rays_numsteps n_rays_per_batch x 2 [step ,base]
        # return
        # rgb_output n_rays_per_batch x 3
        self.num_elements = network_output.shape[0]
        self.density_grid_mean = density_grid_mean.detach()
        self.network_output = network_output.detach()
        self.rays_numsteps = rays_numsteps.detach()
        self.rays_numsteps_compacted = rays_numsteps_compacted.detach()
        self.coords_in = coords_in.detach()
        self.n_rays_per_batch=rays_numsteps.shape[0]
        rgb_output = jt.code((self.n_rays_per_batch, 3), 'float32',
                             inputs=[network_output, coords_in, rays_numsteps, rays_numsteps_compacted,training_background_color], 
                             cuda_header=global_headers+self.density_grad_header+'#include "calc_rgb.h"', cuda_src=f"""
        #define grad_t in0_type
        @alias(network_output, in0)
        @alias(coords_in, in1)
        @alias(rays_numsteps,in2)
        @alias(rgb_output,out0)
        @alias(rays_numsteps_compacted,in3)
        @alias(training_background_color,in4)
        cudaStream_t stream=0;
    
     
        const unsigned int num_elements=network_output_shape0;
        const uint32_t n_rays=rays_numsteps_shape0;
        BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant({self.aabb_range[0]}), Eigen::Vector3f::Constant({self.aabb_range[1]}));
        uint32_t padded_output_width=network_output_shape1;        
        ENerfActivation rgb_activation=ENerfActivation({self.rgb_activation});
        ENerfActivation density_activation=ENerfActivation({self.density_activation});

        linear_kernel(compute_rgbs<grad_t>, 0,stream,
            n_rays, m_aabb,padded_output_width,(grad_t*)network_output_p,rgb_activation,density_activation,
            PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),(uint32_t*)rays_numsteps_p,(Array3f*)rgb_output_p,(uint32_t*)rays_numsteps_compacted_p,(Array3f*)training_background_color_p,NERF_CASCADES(),MIN_CONE_STEPSIZE());   
""")

        rgb_output.compile_options = self.rgb_options
        rgb_output.sync()
        self.rgb_output = rgb_output.detach()
        return rgb_output

    def grad(self, grad_x):
       # return
       # dloss_doutput num_element x 4
        dloss_doutput = jt.code((self.num_elements, 4), self.grad_type,
                                inputs=[self.network_output, self.rays_numsteps_compacted, self.coords_in, grad_x, self.rgb_output, self.density_grid_mean], 
                                cuda_header=global_headers+self.density_grad_header+'#include "calc_rgb.h"', cuda_src=f"""
        #define grad_t out0_type
        @alias(network_output, in0)
        @alias(rays_numsteps, in1)
        @alias(coords_in,in2)
        @alias(grad_x,in3)
        @alias(rgb_output,in4)
        @alias(density_grid_mean,in5)
        @alias(dloss_doutput,out0)


        cudaStream_t stream=0;
    
        cudaMemsetAsync(out0_p, 0, out0->size);
        const unsigned int num_elements=network_output_shape0;
        const uint32_t n_rays=rays_numsteps_shape0;
        BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant({self.aabb_range[0]}), Eigen::Vector3f::Constant({self.aabb_range[1]}));
        uint32_t padded_output_width=network_output_shape1;
        ENerfActivation rgb_activation=ENerfActivation({self.rgb_activation});
        ENerfActivation density_activation=ENerfActivation({self.density_activation});
        linear_kernel(compute_rgbs_grad<grad_t>, 0,stream,
            n_rays, m_aabb,padded_output_width,(grad_t*)dloss_doutput_p,(grad_t*)network_output_p,(uint32_t*)rays_numsteps_p,
            PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),rgb_activation,density_activation,(Array3f*)grad_x_p,(Array3f*)rgb_output_p,(float*)density_grid_mean_p,NERF_CASCADES(),MIN_CONE_STEPSIZE());   

""")

        dloss_doutput.compile_options=self.rgb_options
        dloss_doutput.sync()
        return dloss_doutput, None, None, None,None

    def inference(self, network_output, coords_in, rays_numsteps, density_grid_mean):
        # input
        # network_output num_elements x 4 fp16 maybe
        # coords_in n_rays_per_batch x 7
        # rays_numsteps n_rays_per_batch x 2 [step ,base]
        # return
        # rgb_output n_rays_per_batch x 3
        self.n_rays_per_batch=rays_numsteps.shape[0]
        rgb_output = jt.empty([self.n_rays_per_batch, 3],'float32')
        alpha_output = jt.empty([self.n_rays_per_batch, 1],'float32')
        rgb_output,alpha_output = jt.code(inputs=[network_output, coords_in, rays_numsteps], outputs=[rgb_output,alpha_output],
                             cuda_header=global_headers+self.density_grad_header+'#include"calc_rgb.h"', cuda_src=f"""
        #define grad_t in0_type
        @alias(network_output, in0)
        @alias(coords_in, in1)
        @alias(rays_numsteps,in2)
        @alias(rgb_output,out0)
        @alias(alpha_output,out1)
 

        cudaStream_t stream=0;
    
     
        const unsigned int num_elements=network_output_shape0;
        const uint32_t n_rays=rays_numsteps_shape0;
        BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant({self.aabb_range[0]}), Eigen::Vector3f::Constant({self.aabb_range[1]}));
        uint32_t padded_output_width=network_output_shape1;

        Array3f bg_color=Array3f( {self.bg_color[0]},{self.bg_color[1]},{self.bg_color[2]} );
        
        ENerfActivation rgb_activation=ENerfActivation({self.rgb_activation});
        ENerfActivation density_activation=ENerfActivation({self.density_activation});
        linear_kernel(compute_rgbs_inference<grad_t>, 0, stream,
            n_rays, m_aabb,padded_output_width,bg_color,(grad_t*)network_output_p,rgb_activation,density_activation, PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),(uint32_t*)rays_numsteps_p,(Array3f*)rgb_output_p,NERF_CASCADES(),MIN_CONE_STEPSIZE(),alpha_output_p);      
""")

        rgb_output.compile_options = self.rgb_options
        rgb_output.sync()
        return rgb_output,alpha_output