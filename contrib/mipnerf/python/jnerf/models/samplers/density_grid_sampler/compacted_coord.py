import os
import jittor as jt
from jittor import Function, exp, log
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
jt.flags.use_cuda = 1

class CompactedCoord(Function):
    def __init__(self, density_grad_header, aabb_range=(-1.5, 2.5), n_rays_per_batch=4096, n_rays_step=1024, using_fp16=False, compacted_elements=None):
        self.density_grad_header=density_grad_header
        self.aabb_range=aabb_range
        self.n_rays_per_batch = n_rays_per_batch
        self.bg_color=[1,1,1]
        self.num_elements = n_rays_per_batch*n_rays_step
    
        if compacted_elements is not None:
            self.compacted_elements=compacted_elements
        else:
            self.compacted_elements=self.num_elements//16
        ##activation 0:None 1:relu 2:sigmoid 3:exp
        self.rgb_activation=2
        self.density_activation=3
        self.grad_type='float32'
        if using_fp16:
            self.grad_type='float16'


    def execute(self, network_output, coords_in, rays_numsteps):
        # input
        # network_output num_elements x 4 fp16 maybe
        # coords_in n_rays_per_batch x 7
        # rays_numsteps n_rays_per_batch x 2 [step ,base]
        # return 
        # rgb_output n_rays_per_batch x 3
        compacted_numstep_counter=jt.zeros([1],'int32')
        compacted_rays_counter=jt.zeros([1],'int32')
        rays_numsteps_compacted=jt.empty(rays_numsteps.shape,'int32')
        coords_out=jt.zeros([self.compacted_elements,7],'float32')
        coords_out,rays_numsteps_compacted ,compacted_rays_counter,compacted_numstep_counter= jt.code(inputs=[ network_output, coords_in,rays_numsteps],outputs=[coords_out,rays_numsteps_compacted,compacted_rays_counter,compacted_numstep_counter], 
        cuda_header=global_headers+self.density_grad_header+'#include "compacted_coord.h"', cuda_src=f"""
        #define grad_t in0_type
        @alias(network_output, in0)
        @alias(coords_in, in1)
        @alias(rays_numsteps,in2)
        @alias(coords_out,out0)
      
        @alias(rays_numsteps_compacted,out1)
        @alias(compacted_rays_counter,out2)
        @alias(compacted_numstep_counter,out3)
        cudaStream_t stream=0;
        const unsigned int compacted_elements=coords_out_shape0;
     
        const uint32_t n_rays=rays_numsteps_shape0;
        BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant({self.aabb_range[0]}), Eigen::Vector3f::Constant({self.aabb_range[1]}));
        uint32_t padded_output_width=network_output_shape1;
        Array4f bg_color=Array4f( {self.bg_color[0]},{self.bg_color[1]},{self.bg_color[2]},1 );
        
        ENerfActivation rgb_activation=ENerfActivation({self.rgb_activation});
        ENerfActivation density_activation=ENerfActivation({self.density_activation});
        linear_kernel(compacted_coord<grad_t>,0,stream,
            n_rays, m_aabb, compacted_elements,padded_output_width,bg_color,(grad_t*)network_output_p,rgb_activation,density_activation,
            (NerfCoordinate*)coords_in_p,(NerfCoordinate*)coords_out_p,(uint32_t*)rays_numsteps_p,(uint32_t*)compacted_numstep_counter_p,(uint32_t*)rays_numsteps_compacted_p,(uint32_t*)compacted_rays_counter_p);
           
""")
        
        coords_out.compile_options =proj_options
        coords_out.sync()
        coords_out=coords_out.detach()
        rays_numsteps=rays_numsteps.detach()
        return coords_out,rays_numsteps_compacted,compacted_numstep_counter

    def grad(self, *args):
        ##should not reach here
        assert(False)
        return None