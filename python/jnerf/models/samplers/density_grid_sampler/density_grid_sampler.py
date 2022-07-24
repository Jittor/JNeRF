import os
import jittor as jt
from jittor import nn
from .ema_grid_samples_nerf import ema_grid_samples_nerf
from .generate_grid_samples_nerf_nonuniform import generate_grid_samples_nerf_nonuniform
from .splat_grid_samples_nerf_max_nearest_neighbor import splat_grid_samples_nerf_max_nearest_neighbor
from .update_bitfield import update_bitfield
from .mark_untrained_density_grid import mark_untrained_density_grid
from .compacted_coord import CompactedCoord
from .ray_sampler import RaySampler
from .calc_rgb import CalcRgb
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import SAMPLERS
from jnerf.ops.code_ops.global_vars import global_headers, proj_options
from math import ceil, log2

@SAMPLERS.register_module()
class DensityGridSampler(nn.Module):
    def __init__(self, update_den_freq=16, update_block_size=5000000):
        super(DensityGridSampler, self).__init__()
        self.cfg = get_cfg()
        self.model = self.cfg.model_obj
        self.dataset = self.cfg.dataset_obj
        self.update_den_freq = update_den_freq
        self.update_block_size = update_block_size
        
        # NERF const param
        self.n_rays_per_batch = self.cfg.n_rays_per_batch  # 4096
        self.cone_angle_constant = self.cfg.cone_angle_constant
        self.using_fp16 = self.cfg.fp16
        self.near_distance = self.cfg.near_distance
        self.n_training_steps = self.cfg.n_training_steps
        self.target_batch_size = self.cfg.target_batch_size
        self.const_dt=self.cfg.const_dt
        self.NERF_CASCADES = 5
        self.NERF_GRIDSIZE = 128
        self.NERF_RENDERING_NEAR_DISTANCE = 0.05
        self.NERF_MIN_OPTICAL_THICKNESS = 0.01
        self.MAX_STEP = 1024
        self.background_color = self.cfg.background_color

        self.n_images = self.dataset.n_images
        self.image_resolutions = self.dataset.resolution
        self.W = self.image_resolutions[0]
        self.H = self.image_resolutions[1]
        self.total_rgb = self.n_images*self.H*self.W
        self.aabb_range = self.dataset.aabb_range
        # train param init
        self.read_rgbs = 0
        self.n_rays_total = 0
        self.padded_output_width = 4
        self.density_mlp_padded_density_output_width = 1
        self.n_threads_linear = 128

        # check aabb_scale
        max_aabb_scale = 1 << (self.NERF_CASCADES - 1)
        if self.dataset.aabb_scale > max_aabb_scale: 
            self.NERF_CASCADES = ceil(log2(self.dataset.aabb_scale)) + 1
            print(f'''Warning:Default max value of NeRF dataset's aabb_scale is {max_aabb_scale}, but now is {self.dataset.aabb_scale}.
            You can increase this max_aabb_scale limit by factors of 2 by incrementing NERF_CASCADES. We automatically help you set NERF_CASCADES to {self.NERF_CASCADES}, which may result in slower speeds.''')
        
        self.max_cascade = 0
        while (1 << self.max_cascade) < self.dataset.aabb_scale:
            self.max_cascade += 1
        # array init
        self.exposure = jt.zeros((self.dataset.n_images*3), 'float32')
        self.numsteps_counter = jt.zeros([self.n_training_steps], 'int32')
        self.rays_counter = jt.zeros([1], 'int32')
        # density grid array init
        self.density_grid_decay = 0.95
        self.density_n_elements = self.NERF_CASCADES * \
            self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*self.NERF_GRIDSIZE
        self.density_grid = jt.zeros([self.density_n_elements], 'float32')
        self.density_grid_tmp = jt.zeros(
            [self.density_n_elements], 'float32')
        self._density_grid_indices = jt.zeros(
            [self.density_n_elements], 'int32')

        self._mlp_out = jt.empty([1])
        self.size_including_mips = self.NERF_GRIDSIZE * \
            self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*self.NERF_CASCADES//8
        self.density_grid_bitfield_n_elements = self.NERF_GRIDSIZE * \
            self.NERF_GRIDSIZE*self.NERF_GRIDSIZE
        self.density_grid_bitfield = jt.zeros(
            [self.size_including_mips], 'uint8')
        self._density_grid_positions = jt.empty([1])
        self.density_grid_mean = jt.zeros([self.div_round_up(
            self.density_grid_bitfield_n_elements, self.n_threads_linear)])
        # self.density_grid_ema_step = 0
        self.density_grid_ema_step = jt.zeros([1], 'int32')
        self.dataset_ray_data = False  # 数据集是否包含光线信息
        
        header_path = os.path.join(os.path.dirname(__file__), 'op_header')
        proj_options[f"FLAGS: -I{header_path}"]=1

        self.density_grad_header = f"""
        inline constexpr __device__ uint32_t NERF_GRIDSIZE() {{ return {self.NERF_GRIDSIZE}; }} // size of the density/occupancy grid.
        inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() {{ return {self.NERF_RENDERING_NEAR_DISTANCE}f; }}
        inline constexpr __device__ uint32_t NERF_STEPS() {{ return {self.MAX_STEP}; }} // finest number of steps per unit length
        inline constexpr __device__ uint32_t NERF_CASCADES() {{ return {self.NERF_CASCADES}; }}
        inline __device__ float NERF_MIN_OPTICAL_THICKNESS() {{ return  {self.NERF_MIN_OPTICAL_THICKNESS}f; }}
        inline constexpr __device__ float SQRT3() {{ return 1.73205080757f; }}
        inline constexpr __device__ float STEPSIZE() {{ return (SQRT3() / NERF_STEPS()); }} // for nerf raymarch
        inline constexpr __device__ float MIN_CONE_STEPSIZE() {{ return STEPSIZE(); }}
        inline constexpr __device__ float MAX_CONE_STEPSIZE() {{ return STEPSIZE() * (1 << (NERF_CASCADES() - 1)) * NERF_STEPS() / NERF_GRIDSIZE(); }}
        """
        if self.const_dt:
            self.density_grad_header+="""
            inline __device__ float calc_dt(float t, float cone_angle){return MIN_CONE_STEPSIZE() * 0.5;}
            """
        else:
            self.density_grad_header+="""
            inline __device__ float clamp_(float val, float lower, float upper){return val < lower ? lower : (upper < val ? upper : val);}
            inline __device__ float calc_dt(float t, float cone_angle){ return clamp_(t * cone_angle, MIN_CONE_STEPSIZE(), MAX_CONE_STEPSIZE());}
            """
        self.density_grad_host_header = self.density_grad_header.replace("__device__", "__device__ __host__")

        self.mark_untrained_density_grid = mark_untrained_density_grid(
            self.density_grad_header, n_images=self.n_images, image_resolutions=self.image_resolutions)
        self.generate_grid_samples_nerf_nonuniform = generate_grid_samples_nerf_nonuniform(
            self.density_grad_header, aabb_range=self.aabb_range)
        self.splat_grid_samples_nerf_max_nearest_neighbor = splat_grid_samples_nerf_max_nearest_neighbor(
            self.density_grad_header, padded_output_width=self.density_mlp_padded_density_output_width, using_fp16=self.using_fp16)
        self.ema_grid_samples_nerf = ema_grid_samples_nerf(
            self.density_grad_header, decay=0.95)
        self.update_bitfield = update_bitfield(
            self.density_grad_header)
        self.rays_sampler = RaySampler(
            self.density_grad_host_header, self.near_distance, self.cone_angle_constant, self.aabb_range, self.n_rays_per_batch, self.MAX_STEP)
        self.compacted_coords = CompactedCoord(
            self.density_grad_host_header, self.aabb_range, self.n_rays_per_batch, self.MAX_STEP, self.using_fp16, self.target_batch_size)
        self.calc_rgb = CalcRgb(self.density_grad_host_header, self.aabb_range, self.n_rays_per_batch, self.MAX_STEP, self.padded_output_width, self.background_color, using_fp16=self.using_fp16)

        self.measured_batch_size=jt.zeros([1],'int32')##rays batch sum

    def sample(self, img_ids, rays_o, rays_d, rgb_target=None, is_training=False):
        if is_training:
            if self.cfg.m_training_step%self.update_den_freq==0:
                self.update_density_grid()
        coords, rays_index, rays_numsteps, rays_numsteps_counter = self.rays_sampler.execute(
            rays_o=rays_o, rays_d=rays_d, density_grid_bitfield=self.density_grid_bitfield,
            metadata=self.dataset.metadata, imgs_id=img_ids, xforms=self.dataset.transforms_gpu)
        coords_pos = coords[...,  :3].detach()
        coords_dir = coords[..., 4: ].detach()
        if not is_training:
            self._coords = coords.detach()
            self._rays_numsteps = rays_numsteps.detach()
            return coords_pos, coords_dir

        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                nerf_outputs = self.model(coords_pos, coords_dir).detach()
                coords_compacted,rays_numsteps_compacted,compacted_numstep_counter=self.compacted_coords(nerf_outputs,coords,rays_numsteps)
                self.measured_batch_size+=compacted_numstep_counter
        else:
            nerf_outputs = self.model(coords_pos, coords_dir).detach()
            coords_compacted,rays_numsteps_compacted,compacted_numstep_counter=self.compacted_coords(nerf_outputs,coords,rays_numsteps)
            self.measured_batch_size+=compacted_numstep_counter
        if is_training:
            if self.cfg.m_training_step%self.update_den_freq==(self.update_den_freq-1):
                self.update_batch_rays()
        coords_compacted=coords_compacted.detach()
        self._coords = coords_compacted.detach()
        self._rays_numsteps = rays_numsteps.detach()
        self._rays_numsteps_compacted = rays_numsteps_compacted.detach()
        return coords_compacted[..., :3].detach(), coords_compacted[..., 4:].detach()
    
    def rays2rgb(self, network_outputs, training_background_color=None, inference=False):
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.rays2rgb_(network_outputs, training_background_color, inference)
        else:
            return self.rays2rgb_(network_outputs, training_background_color, inference)
    
    def rays2rgb_(self, network_outputs, training_background_color=None, inference=False):
        if training_background_color is None:
            background_color = self.background_color
        else:
            background_color = training_background_color
        assert network_outputs.shape[0]==self._coords.shape[0]
        if inference:
            return self.calc_rgb.inference(
                network_outputs, 
                self._coords, 
                self._rays_numsteps,
                self.density_grid_mean)
        else:
            return self.calc_rgb(
                network_outputs,
                self._coords,
                self._rays_numsteps,
                self.density_grid_mean,
                self._rays_numsteps_compacted,
                background_color
            )

    def enlarge(self, x: jt.Var, size: int):
        if x.shape[0] < size:
            y = jt.empty([size], x.dtype)
            x.assign(y)

    def update_density_grid_nerf(self, decay: float, n_uniform_density_grid_samples: int, n_nonuniform_density_grid_samples: int):
        n_elements = self.density_n_elements
        n_density_grid_samples = n_uniform_density_grid_samples + \
            n_nonuniform_density_grid_samples
        self.enlarge(self._density_grid_positions, n_density_grid_samples)
        padded_output_width = self.density_mlp_padded_density_output_width
        self.enlarge(self._mlp_out, n_density_grid_samples*padded_output_width)
        if self.cfg.m_training_step == 0:
            if not self.dataset_ray_data:
                self.density_grid = self.mark_untrained_density_grid(
                    self.dataset.focal_lengths, self.dataset.transforms_gpu, n_elements)
            else:
                jt.init.zero_(self.density_grid)

        jt.init.zero_(self.density_grid_tmp)
        density_grid_positions_uniform, density_grid_indices_uniform = self.generate_grid_samples_nerf_nonuniform.execute(
            self.density_grid, n_uniform_density_grid_samples, self.density_grid_ema_step, self.max_cascade, -0.01)
        density_grid_positions_nonuniform, density_grid_indices_nonuniform = self.generate_grid_samples_nerf_nonuniform.execute(
            self.density_grid, n_nonuniform_density_grid_samples, self.density_grid_ema_step, self.max_cascade, self.NERF_MIN_OPTICAL_THICKNESS)
        self._density_grid_positions = jt.concat(
            [density_grid_positions_uniform, density_grid_positions_nonuniform])
        self._density_grid_indices = jt.concat(
            [density_grid_indices_uniform, density_grid_indices_nonuniform])
        self._density_grid_positions = self._density_grid_positions.reshape(
            -1, 3)
        with jt.no_grad():
            bs = self.update_block_size
            res=[]
            for i in range(0,self._density_grid_positions.shape[0],bs):
                if self.using_fp16:
                    with jt.flag_scope(auto_mixed_precision_level=5):
                        res.append(self.model.density(self._density_grid_positions[i:i+bs]))
                else:
                    res.append(self.model.density(self._density_grid_positions[i:i+bs]))
            self._mlp_out = jt.concat(res,0)
        self.density_grid_tmp = self.splat_grid_samples_nerf_max_nearest_neighbor.execute(
            self._density_grid_indices, self._mlp_out, self.density_grid_tmp, n_density_grid_samples)
  
        self.density_grid = self.ema_grid_samples_nerf.execute(
             self.density_grid_tmp, self.density_grid, n_elements)
      
        self.density_grid = self.density_grid.detach()
        self.density_grid_ema_step += 1
        ## update_density_grid_mean_and_bitfield function
        self.density_grid_mean = jt.zeros_like(self.density_grid_mean)
        self.density_grid_bitfield ,self.density_grid_mean= self.update_bitfield.execute(
            self.density_grid, self.density_grid_mean, self.density_grid_bitfield)

    def div_round_up(self, val, divisor):
        return (val+divisor-1) // divisor

    def update_density_grid(self):
        alpha = pow(self.density_grid_decay, self.n_training_steps / 16)
        n_cascades = self.max_cascade+1
        if self.cfg.m_training_step < 256:
            self.update_density_grid_nerf(
                alpha, self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*n_cascades, 0)
        else:
            self.update_density_grid_nerf(alpha, self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*self.NERF_GRIDSIZE *
                                        n_cascades//4, self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*n_cascades//4)
        jt.gc()

    def update_batch_rays(self):
        measured_batch_size=max(self.measured_batch_size.item()/16,1)
        rays_per_batch=int(self.n_rays_per_batch*self.target_batch_size/measured_batch_size)
        self.n_rays_per_batch=int(min(self.div_round_up(int(rays_per_batch),128)*128,self.target_batch_size))
        jt.init.zero_(self.measured_batch_size)
        self.dataset.batch_size=self.n_rays_per_batch
    
    def set_status(self, is_train):
        if not is_train:
            self.n_rays_per_batch = self.cfg.n_rays_per_batch_test
        else:
            self.n_rays_per_batch = self.cfg.n_rays_per_batch
        self.compacted_coords.set_status(is_train, self.n_rays_per_batch)
        self.calc_rgb.set_status(is_train, self.n_rays_per_batch)
        self.rays_sampler.set_status(is_train, self.n_rays_per_batch)
    
    def bbox_test(self, rays_o, rays_d, metadata, xforms, n_rays, img_id):
        cuda_header = '''
        #include "ray_sampler.h"
        '''
        cuda_header = global_headers + self.density_grad_header + cuda_header
        cuda_src = f'''
        @alias(rays_o, in0)
        @alias(rays_d, in1)
        @alias(metadata, in2)
        @alias(training_xforms, in3)
        @alias(img_id, in4)
        @alias(cone_angles, out0)
        @alias(hits, out1)
        cudaStream_t stream = 0;
        float cone_angle_constant = {self.cone_angle_constant};
        float near_distance = {self.near_distance};
        uint32_t n_rays = cone_angles_shape0;
        BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant({self.aabb_range[0]}), Eigen::Vector3f::Constant({self.aabb_range[1]}));
        linear_kernel(get_hits_t, 0, stream, n_rays, (Vector3f*) rays_o_p, (Vector3f*) rays_d_p, cone_angle_constant, m_aabb, (uint32_t*)img_id_p, near_distance, (TrainingImageMetadata *)metadata_p, (Eigen::Matrix<float, 3, 4>*) training_xforms_p, (float*)cone_angles_p, (float*) hits_p, rng);
        rng.advance();
        '''
        cone_angles = jt.zeros((rays_o.shape[0])).float32()
        hits = jt.zeros((rays_o.shape[0], 2)).float32()
        jt.sync_all()
        cone_angles, hits = jt.code([rays_o, rays_d, metadata, xforms, img_id], [cone_angles, hits], cuda_header=cuda_header, cuda_src=cuda_src)
        cone_angles.compile_options = proj_options
        return cone_angles, hits
    
    def test_raysample(self, rays_o, rays_d, hits, cone_angles, numsteps_in, metadata, alive_indices, n_rays, img_id, spr=1):
        cuda_header = '''
        #include "ray_sampler.h"
        '''
        cuda_header = global_headers + self.density_grad_header + cuda_header
        cuda_src = f'''
        @alias(rays_o, in0)
        @alias(rays_d, in1)
        @alias(density_grid, in2)
        @alias(metadata, in3)
        @alias(numsteps_in, in4)
        @alias(cone_angles, in5)
        @alias(img_id, in6)
        @alias(coords_out, out0)
        @alias(N_eff_samples, out1)
        @alias(alive_indices, out2)
        @alias(hits, out3)
        cudaStream_t stream = 0;
        uint32_t n_rays = N_eff_samples_shape0;
        uint32_t spr = coords_out_shape0 / n_rays;
        BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant({self.aabb_range[0]}), Eigen::Vector3f::Constant({self.aabb_range[1]}));
        linear_kernel(raymarching_test_kernel, 0, stream, n_rays, (Vector3f *)rays_o_p, (Vector3f *)rays_d_p, (Vector2f *)hits_p, m_aabb, (float*)cone_angles_p, (int*)alive_indices_p, (uint8_t*)density_grid_p, spr, (uint32_t*)img_id_p, (TrainingImageMetadata *)metadata_p, (uint32_t*)numsteps_in_p, PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_out_p, 1, 0, 0), (int*)N_eff_samples_p);
        '''
        N_alive = alive_indices.shape[0]
        coords_out = jt.empty((spr * N_alive, 7), 'float32')
        N_eff_samples = jt.zeros((alive_indices.shape[0])).int32()
        coords, N_eff_samples, alive_indices, hits = jt.code([rays_o, rays_d, self.density_grid_bitfield, metadata, numsteps_in, cone_angles, img_id], [coords_out, N_eff_samples, alive_indices, hits], cuda_header=cuda_header,cuda_src=cuda_src)
        coords_out.compile_options = proj_options
        return coords, N_eff_samples, hits

    def additional_calc_rgb(self, network_outputs, coords_in, rgbs, opacity, numsteps_in, N_eff_samples, alive_indices, n_rays):
        return self.calc_rgb.additional_calc_rgb(network_outputs, coords_in, rgbs, opacity, numsteps_in, N_eff_samples, alive_indices, n_rays)
