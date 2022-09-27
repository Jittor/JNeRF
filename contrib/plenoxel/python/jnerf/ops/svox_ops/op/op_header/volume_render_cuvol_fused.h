#include <stdio.h>
#include "cuda_util.h"
#include "data_spec.h"
#include "jt_helper.h"
#include "render_util.cuh"
const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 8;

const int TRACE_RAY_BG_CUDA_THREADS = 128;
const int MIN_BG_BLOCKS_PER_SM = 8;

typedef cub::WarpReduce<float> WarpReducef;

template <typename T, size_t D>
__global__ void test_packed_var(PackedVar<T, D> pvar) {
    int num = pvar._num;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > num)
        return;
    printf("num %d\n", tid);
    pvar(tid) = tid;
}
__global__ void test(Var* x) {
    int num = x->num;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > num)
        return;

    int num11 = x->num;
    printf("num:%d\n", 10);
    return;
}
// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
    const PackedSparseGridSpec& __restrict__ grid,
    SingleRaySpec& __restrict__ ray,
    const RenderOptions& __restrict__ opt,
    uint32_t lane_id,
    float* __restrict__ sphfunc_val,
    WarpReducef::TempStorage& __restrict__ temp_storage,
    float* __restrict__ output,
    float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    if (ray.tmin > ray.tmax) {
        output[lane_colorgrp] = (grid.background_nlayers == 0) ? opt.background_brightness : 0.f;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }
// for(int i=0;i<1000;i++){
// printf("%d %d %f\n",i,grid.links[i],grid.density_data[i]);
// }
    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        ////printf("line:%d\n", __LINE__);
        const float skip = compute_skip_dist(ray,
                                             grid.links, grid.stride_x,
                                             grid.size[2], 0);

        ////printf("line:%d\n", __LINE__);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        ////printf("line:%d\n", __LINE__);
        // for (int i = 0; i < 3; i++) {
        //     printf(" ray.l %d %d ", i, ray.l[i]);
        // }
        // for (int i = 0; i < 3; i++) {
        //     printf(" ray.pos %d %f ", i, ray.pos[i]);
        // }
        // printf(" %d %d %d \n",grid.stride_x,grid.size[2],1);
        float sigma = trilerp_cuvol_one(
            grid.links, grid.density_data,
            grid.stride_x,
            grid.size[2],
            1,
            ray.l, ray.pos,
            0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        ////printf("line:%d\n", __LINE__);
        // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                grid.links,
                grid.sh_data,
                grid.stride_x,
                grid.size[2],
                grid.sh_data_dim,
                ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id];  // bank conflict

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        // printf("t skip outv sigma %f  %f %f %f\n", t, skip, outv, sigma);
        ////printf("line:%d\n", __LINE__);
        t += opt.step_size;
    }
    // printf("line:%d %f \n", __LINE__, outv);
    ////printf("line:%d\n", __LINE__);
    if (grid.background_nlayers == 0) {
        outv += _EXP(log_transmit) * opt.background_brightness;
    }
    // printf("line:%d %f \n", __LINE__, outv);

    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        ////printf("line:%d\n", __LINE__);
        // printf("lane_colorgrp:%d\n", lane_colorgrp);
        output[lane_colorgrp] = outv;
    }
    ////printf("line:%d\n", __LINE__);
}

__global__ void render_ray_kernel(PackedSparseGridSpec grid,
                                  PackedRaysSpec rays,
                                  RenderOptions opt,
                                  PackedVar32<float, 2> out,
                                  float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    // TODOdy:recover
    if (lane_id >= grid.sh_data_dim)  // Bad, but currently the best way due to coalesced memory access
        return;

    // if (ray_blk_id != 1 || lane_id != 18 || ray_id != 1 || (tid & 0x1F != 1))
    //     return;
    // printf("id %d %d %d %d\n", tid, ray_id, ray_blk_id, lane_id);
    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    // // TODOdy:remove
    // if (tid != 0)
    //     return;
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    // printf("ray_spec %f %f %f %f %f %f\n", ray_spec[ray_blk_id].origin[0], ray_spec[ray_blk_id].origin[1], ray_spec[ray_blk_id].origin[2],
        //    ray_spec[ray_blk_id].dir[0], ray_spec[ray_blk_id].dir[1], ray_spec[ray_blk_id].dir[2]);
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    // printf("sphfunc_val ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%f ", sphfunc_val[ray_blk_id][i]);
    // }
    // printf("\n");
    // TODOdy:opt
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    // printf("ray spec tmin tmax %f %f\n", ray_spec[ray_blk_id].tmin, ray_spec[ray_blk_id].tmax);
    __syncwarp((1U << grid.sh_data_dim) - 1);
    // printf("lane_id%d\n", lane_id);
    // printf("ray id %d\n", ray_id);
    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out[ray_id].data(),
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
    // for (int i = 0; i < 3; i++) {
    //     printf("out %d %f ", i, out[ray_id].data()[i]);
    // }
    // rays.dirs(tid) = tid;
    return;
}
__device__ __inline__ void trace_ray_cuvol_backward(
    const PackedSparseGridSpec& __restrict__ grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    SingleRaySpec& __restrict__ ray,
    const RenderOptions& __restrict__ opt,
    uint32_t lane_id,
    const float* __restrict__ sphfunc_val,
    float* __restrict__ grad_sphfunc_val,
    WarpReducef::TempStorage& __restrict__ temp_storage,
    float log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads& __restrict__ grads,
    float* __restrict__ accum_out,
    float* __restrict__ log_transmit_out) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim));
    // printf("line:%d\n", __LINE__);
    float accum = fmaf(color_cache[0], grad_output[0],
                       fmaf(color_cache[1], grad_output[1],
                            color_cache[2] * grad_output[2]));

    if (beta_loss > 0.f) {
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3));  // d beta_loss / d log_transmit_in
        accum += beta_loss;
        // Interesting how this loss turns out, kinda nice?
    }
    // printf("line:%d\n", __LINE__);
    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) {
            *accum_out = accum;
        }
        if (log_transmit_out != nullptr) {
            *log_transmit_out = 0.f;
        }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }
    float t = ray.tmin;
    // printf("line:%d\n", __LINE__);
    const float gout = grad_output[lane_colorgrp];
    // printf("line:%d\n", __LINE__);
    float log_transmit = 0.f;

    // remat samples
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray,
                                             grid.links, grid.stride_x,
                                             grid.size[2], 0);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        // printf("line:%d\n", __LINE__);
        float sigma = trilerp_cuvol_one(
            grid.links,
            grid.density_data,
            grid.stride_x,
            grid.size[2],
            1,
            ray.l, ray.pos,
            0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
        // printf("line:%d\n", __LINE__);
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                grid.links,
                grid.sh_data,
                grid.stride_x,
                grid.size[2],
                grid.sh_data_dim,
                ray.l, ray.pos, lane_id);
            // printf("line:%d\n", __LINE__);
            // printf("lane_colorgrp_id:%d\n", lane_colorgrp_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];
            // printf("line:%d\n", __LINE__);
            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout;  // Clamp to [+0, infty)

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
            total_color += total_color_c1;
            // printf("line:%d\n", __LINE__);
            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

            if (grid.basis_type != BASIS_TYPE_SH) {
                float curr_grad_sphfunc = lane_color * grad_common;
                const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                                                             curr_grad_sphfunc, 2 * grid.basis_dim);
                curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                                                      curr_grad_sphfunc, grid.basis_dim);
                curr_grad_sphfunc += curr_grad_up2;
                if (lane_id < grid.basis_dim) {
                    grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                }
            }
            // printf("line:%d\n", __LINE__);
            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                // Cauchy version (from SNeRG)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                // Alphs version (from PlenOctrees)
                // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }
            // printf("line:%d\n", __LINE__);
            // printf("lane_id%d\n", lane_id);
            trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                                       grid.stride_x,
                                       grid.size[2],
                                       grid.sh_data_dim,
                                       ray.l, ray.pos,
                                       curr_grad_color, lane_id);
            // printf("line:%d\n", __LINE__);
            if (lane_id == 0) {

                trilerp_backward_cuvol_one_density(
                    grid.links,
                    grads.grad_density_out,
                    grads.mask_out,
                    grid.stride_x,
                    grid.size[2],
                    ray.l, ray.pos, curr_grad_sigma);
            }
            // printf("line:%d\n", __LINE__);
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        // printf("line:%d\n", __LINE__);
        t += opt.step_size;
    }
    // printf("line:%d\n", __LINE__);
    if (lane_id == 0) {
        if (accum_out != nullptr) {
            // Cancel beta loss out in case of background
            accum -= beta_loss;
            *accum_out = accum;
        }
        if (log_transmit_out != nullptr) {
            *log_transmit_out = log_transmit;
        }
        // printf("accum_end_fg=%f\n", accum);
        // printf("log_transmit_fg=%f\n", log_transmit);
    }
}
__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr,
    float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;
    //  if (ray_blk_id != 1 || lane_id != 0 || ray_id != 1 || (tid & 0x1F != 1))
    //     return;
    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                           ray_spec[ray_blk_id].dir[1],
                           ray_spec[ray_blk_id].dir[2]};
    // printf("vdir %f %f %f\n",vdir[0],vdir[1],vdir[2]);
    if (lane_id < grid.basis_dim) {
        grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
    }
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 vdir, sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }
    float grad_out[3];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }
    // printf("grad_out %f %f %f\n",grad_out[0],grad_out[1],grad_out[2]);
    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_cuvol_backward(
        grid,
        grad_out,
        color_cache + ray_id * 3,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        grads,
        accum_out == nullptr ? nullptr : accum_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
    // printf("line:%d\n", __LINE__);
    calc_sphfunc_backward(
        grid, lane_id,
        ray_id,
        vdir,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        grads.grad_basis_out);
        // grads.grad_density_out[0]=1;
        // grads.grad_sh_out[0]=2;
    // printf("line:%d\n", __LINE__);
}

__device__ __inline__ void render_background_forward(
            const PackedSparseGridSpec& __restrict__ grid,
            SingleRaySpec& __restrict__ ray,
            const RenderOptions& __restrict__ opt,
            float log_transmit,
            float* __restrict__ out
        ) {

    ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    float t, invr_last = 1.f / inner_radius;
    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    // csi.intersect(inner_radius, &t_last);

    float outv[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < n_steps; ++i) {
        // Between 1 and infty
        float r = n_steps / (n_steps - i - 0.5);
        if (r < inner_radius || !csi.intersect(r, &t)) continue;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
        }
        const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] *= invr_mid;
        }
        // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
        _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
        ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                       grid.background_nlayers - 1);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.l[j] = (int) ray.pos[j];
        }
        ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
        ray.l[1] = min(ray.l[1], grid.background_reso - 1);
        ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] -= ray.l[j];
        }

        float sigma = trilerp_bg_one(
                grid.background_links,
                grid.background_data,
                grid.background_reso,
                grid.background_nlayers,
                4,
                ray.l,
                ray.pos,
                3);

        // if (i == n_steps - 1) {
        //     ray.world_step = 1e9;
        // }
        // if (opt.randomize && opt.random_sigma_std_background > 0.0)
        //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > 0.f) {
            const float pcnt = (invr_last - invr_mid) * ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;
#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                // Not efficient
                const float color = trilerp_bg_one(
                        grid.background_links,
                        grid.background_data,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        i) * C0;  // Scale by SH DC factor to help normalize lrs
                outv[i] += weight * fmaxf(color + 0.5f, 0.f);  // Clamp to [+0, infty)
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        invr_last = invr_mid;
    }
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        out[i] += outv[i] + _EXP(log_transmit) * opt.background_brightness;
    }
}

__device__ __inline__ void render_background_backward(
            const PackedSparseGridSpec& __restrict__ grid,
            const float* __restrict__ grad_output,
            SingleRaySpec& __restrict__ ray,
            const RenderOptions& __restrict__ opt,
            float log_transmit,
            float accum,
            float sparsity_loss,
            PackedGridOutputGrads& __restrict__ grads
        ) {
    // printf("accum_init=%f\n", accum);
    // printf("log_transmit_init=%f\n", log_transmit);
    ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    float t, invr_last = 1.f / inner_radius;
    // csi.intersect(inner_radius, &t_last);
    for (int i = 0; i < n_steps; ++i) {
        float r = n_steps / (n_steps - i - 0.5);

        if (r < inner_radius || !csi.intersect(r, &t)) continue;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
        }

        const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] *= invr_mid;
        }
        // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
        _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
        ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                       grid.background_nlayers - 1);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.l[j] = (int) ray.pos[j];
        }
        ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
        ray.l[1] = min(ray.l[1], grid.background_reso - 1);
        ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] -= ray.l[j];
        }


        float sigma = trilerp_bg_one(
                grid.background_links,
                grid.background_data,
                grid.background_reso,
                grid.background_nlayers,
                4,
                ray.l,
                ray.pos,
                3);
        // if (i == n_steps - 1) {
        //     ray.world_step = 1e9;
        // }

        // if (opt.randomize && opt.random_sigma_std_background > 0.0)
        //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > 0.f) {
            float total_color = 0.f;
            const float pcnt = ray.world_step * (invr_last - invr_mid) * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            for (int i = 0; i < 3; ++i) {
                const float color = trilerp_bg_one(
                        grid.background_links,
                        grid.background_data,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        i) * C0 + 0.5f;  // Scale by SH DC factor to help normalize lrs

                total_color += fmaxf(color, 0.f) * grad_output[i];
                if (color > 0.f) {
                    const float curr_grad_color = C0 * weight * grad_output[i];
                    trilerp_backward_bg_one(
                            grid.background_links,
                            grads.grad_background_out,
                            nullptr,
                            grid.background_reso,
                            grid.background_nlayers,
                            4,
                            ray.l,
                            ray.pos,
                            curr_grad_color,
                            i);
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (invr_last - invr_mid) * (
                    total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                // Cauchy version (from SNeRG)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                // Alphs version (from PlenOctrees)
                // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }

            trilerp_backward_bg_one(
                    grid.background_links,
                    grads.grad_background_out,
                    grads.mask_background_out,
                    grid.background_reso,
                    grid.background_nlayers,
                    4,
                    ray.l,
                    ray.pos,
                    curr_grad_sigma,
                    3);

            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        invr_last = invr_mid;
    }
}
__launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        // Outputs
        PackedVar32<float,2> out) {
    CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
    if (log_transmit[ray_id] < -25.f) return;
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
    render_background_forward(
        grid,
        ray_spec,
        opt,
        log_transmit[ray_id],
        out[ray_id].data());
}



__launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_backward_kernel(
        PackedSparseGridSpec grid,
        const float* __restrict__ grad_output,
        const float* __restrict__ color_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        const float* __restrict__ accum,
        bool grad_out_is_rgb,
        float sparsity_loss,
        // Outputs
        PackedGridOutputGrads grads) {
    CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
    if (log_transmit[ray_id] < -25.f) return;
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds_bg(ray_spec, grid, opt, ray_id);

    float grad_out[3];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    render_background_backward(
        grid,
        grad_out,
        ray_spec,
        opt,
        log_transmit[ray_id],
        accum[ray_id],
        sparsity_loss,
        grads);
}