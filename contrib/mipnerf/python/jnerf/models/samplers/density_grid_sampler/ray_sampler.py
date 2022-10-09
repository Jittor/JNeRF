import os
import jittor as jt
from jittor import Function, exp, log
import numpy as np
import sys
from jnerf.ops.code_ops.global_vars import global_headers, proj_options
jt.flags.use_cuda = 1

class RaySampler(Function):
    def __init__(self, density_grad_header, near_distance, cone_angle_constant, aabb_range=(-1.5, 2.5), n_rays_per_batch=4096, n_rays_step=1024):
        self.density_grad_header = density_grad_header
        self.aabb_range = aabb_range
        self.near_distance = near_distance
        self.n_rays_per_batch = n_rays_per_batch
        self.num_elements = n_rays_per_batch*n_rays_step
        self.cone_angle_constant = cone_angle_constant
        self.path = os.path.join(os.path.dirname(__file__), '..', 'op_include')
        self.ray_numstep_counter = jt.zeros([2], 'int32')

    def execute(self, rays_o, rays_d, density_grid_bitfield, metadata, imgs_id, xforms):
        # input
        # rays_o n_rays_per_batch x 3
        # rays_d n_rays_per_batch x 3
        # bitfield 128 x 128 x 128 x 5 / 8
        # return
        # coords_out=[self.num_elements,7]
        # rays index : store rays is used ( not for -1)
        # rays_numsteps [0:step,1:base]
        jt.init.zero_(self.ray_numstep_counter)
        coords_out = jt.empty((self.num_elements, 7), 'float32')
        self.n_rays_per_batch=rays_o.shape[0]
        rays_index = jt.empty((self.n_rays_per_batch, 1), 'int32')
        rays_numsteps = jt.empty((self.n_rays_per_batch, 2), 'int32')
        coords_out, rays_index, rays_numsteps,self.ray_numstep_counter = jt.code(
            inputs=[rays_o, rays_d, density_grid_bitfield, metadata, imgs_id, xforms], outputs=[coords_out,rays_index,rays_numsteps,self.ray_numstep_counter], 
            cuda_header=global_headers+self.density_grad_header+'#include "ray_sampler.h"',  cuda_src=f"""
     
        @alias(rays_o, in0)
        @alias(rays_d, in1)
        @alias(density_grid_bitfield,in2)
        @alias(metadata,in3)
        @alias(imgs_index,in4)
        @alias(xforms_input,in5)
        @alias(ray_numstep_counter,out3)
        @alias(coords_out,out0)
        @alias(rays_index,out1)
        @alias(rays_numsteps,out2)

        cudaStream_t stream=0;
        cudaMemsetAsync(coords_out_p, 0, coords_out->size);
     
        const unsigned int num_elements=coords_out_shape0;
        const uint32_t n_rays=rays_o_shape0;
        BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant({self.aabb_range[0]}), Eigen::Vector3f::Constant({self.aabb_range[1]}));
        float near_distance = {self.near_distance};
        float cone_angle_constant={self.cone_angle_constant};  
        linear_kernel(rays_sampler,0,stream,
            n_rays, m_aabb, num_elements,(Vector3f*)rays_o_p,(Vector3f*)rays_d_p, (uint8_t*)density_grid_bitfield_p,cone_angle_constant,(TrainingImageMetadata *)metadata_p,(uint32_t*)imgs_index_p,
            (uint32_t*)ray_numstep_counter_p,((uint32_t*)ray_numstep_counter_p)+1,(uint32_t*)rays_index_p,(uint32_t*)rays_numsteps_p,PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_out_p, 1, 0, 0),(Eigen::Matrix<float, 3, 4>*) xforms_input_p,near_distance,rng);   

        rng.advance();
""")

        coords_out.compile_options = proj_options
        coords_out.sync()
        coords_out = coords_out.detach()
        rays_index = rays_index.detach()
        rays_numsteps = rays_numsteps.detach()
        self.ray_numstep_counter = self.ray_numstep_counter.detach()
        samples=self.ray_numstep_counter[1].item()
        coords_out=coords_out[:samples]
        return coords_out, rays_index, rays_numsteps, self.ray_numstep_counter

    def grad(self, grad_x):
        ##should not reach here
        assert(grad_x == None)
        assert(False)
        return None
