#pragma once
#include <stdio.h>
#include "cuda_util.h"
#include "data_spec.h"
#include "jt_helper.h"
#include "render_util.cuh"

const int MISC_CUDA_THREADS = 256;
const int MISC_MIN_BLOCKS_PER_SM = 4;

__launch_bounds__(MISC_CUDA_THREADS, MISC_MIN_BLOCKS_PER_SM)
    __global__ void dilate_kernel(
        const PackedVar32<bool, 3> grid,
        // Output
        PackedVar32<bool, 3> out_grid) {
    CUDA_GET_THREAD_ID(tid, grid.size(0) * grid.size(1) * grid.size(2));

    const int z = tid % grid.size(2);
    const int xy = tid / grid.size(2);
    const int y = xy % grid.size(1);
    const int x = xy / grid.size(1);

    int xl = max(x - 1, 0), xr = min(x + 1, (int)grid.size(0) - 1);
    int yl = max(y - 1, 0), yr = min(y + 1, (int)grid.size(1) - 1);
    int zl = max(z - 1, 0), zr = min(z + 1, (int)grid.size(2) - 1);

    out_grid[x][y][z] =
        grid[xl][yl][zl] | grid[xl][yl][z] | grid[xl][yl][zr] |
        grid[xl][y][zl] | grid[xl][y][z] | grid[xl][y][zr] |
        grid[xl][yr][zl] | grid[xl][yr][z] | grid[xl][yr][zr] |

        grid[x][yl][zl] | grid[x][yl][z] | grid[x][yl][zr] |
        grid[x][y][zl] | grid[x][y][z] | grid[x][y][zr] |
        grid[x][yr][zl] | grid[x][yr][z] | grid[x][yr][zr] |

        grid[xr][yl][zl] | grid[xr][yl][z] | grid[xr][yl][zr] |
        grid[xr][y][zl] | grid[xr][y][z] | grid[xr][y][zr] |
        grid[xr][yr][zl] | grid[xr][yr][z] | grid[xr][yr][zr];
}

// Fast single-channel rendering for weight-thresholding
__device__ __inline__ void grid_trace_ray(
    // const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
    const PackedVar32<float, 3>
        data,
    SingleRaySpec ray,
    const float* __restrict__ offset,
    const float* __restrict__ scaling,
    float step_size,
    float stop_thresh,
    bool last_sample_opaque,
    // torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
    PackedVar32<float, 3>
        grid_weight,
        int tid) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, scaling, offset);
    // Warning: modifies ray.dir
    const float world_step = _get_delta_scale(scaling, ray.dir) * step_size;

    float t, tmax;
    {
        float t1, t2;
        t = 0.0f;
        tmax = 2e3f;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float invdir = 1.0 / ray.dir[i];
            t1 = (-0.5f - ray.origin[i]) * invdir;
            t2 = (data.size(i) - 0.5f - ray.origin[i]) * invdir;
            if (ray.dir[i] != 0.f) {
                t = max(t, min(t1, t2));
                tmax = min(tmax, max(t1, t2));
            }
        }
    }
    //debug
    // if(tid==5535){
    //     printf("t tmax %f %f \n",t ,tmax);
    // }
    if (t > tmax) {
        // Ray doesn't hit box
        return;
    }
    float pos[3];
    int32_t l[3];

    float log_light_intensity = 0.f;
    const int stride0 = data.size(1) * data.size(2);
    const int stride1 = data.size(2);
    while (t <= tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            pos[j] = ray.origin[j] + t * ray.dir[j];
            pos[j] = min(max(pos[j], 0.f), data.size(j) - 1.f);
            l[j] = (int32_t)pos[j];
            l[j] = min(l[j], data.size(j) - 2);
            pos[j] -= l[j];
        }
        
        float log_att;
        const int idx = l[0] * stride0 + l[1] * stride1 + l[2];

        float sigma;
        {
            // Trilerp
            const float* __restrict__ sigma000 = data.data() + idx;
            const float* __restrict__ sigma100 = sigma000 + stride0;
            const float ix0y0 = lerp(sigma000[0], sigma000[1], pos[2]);
            const float ix0y1 = lerp(sigma000[stride1], sigma000[stride1 + 1], pos[2]);
            const float ix1y0 = lerp(sigma100[0], sigma100[1], pos[2]);
            const float ix1y1 = lerp(sigma100[stride1], sigma100[stride1 + 1], pos[2]);
            const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
            const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
            sigma = lerp(ix0, ix1, pos[0]);
//             if(tid==5535&&idx==36437468){
// printf("sigmaxy %f %f %f %f %f %f %f %f %f\n",ix0y0,ix0y1,ix1y0,ix1y1,ix0,ix1,sigma,sigma000[0],sigma000[1]);
//             }
        }
        if (last_sample_opaque && t + step_size > tmax) {
            sigma += 1e9;
            log_light_intensity = 0.f;
        }

        // const float weight_p = _EXP(log_light_intensity) * (1.f - _EXP(log_att));
        // int s_idx=494;
        // if (tid==5535) {
        //     // if(idx == s_idx || idx + stride1 == s_idx || idx + stride0 == s_idx || idx + stride0 + stride1 == s_idx){
        //     printf("pos l %f %f %f %d %d %d\n",pos[0],pos[1],pos[2],l[0],l[1],l[2]);
        //     printf("weight_p%f sigma %f target %f tid%d log_light_intensity%f log_att%f idx%d\n", weight_p, sigma,grid_weight.data()[s_idx],tid,log_light_intensity,log_att,idx);
        //     // }
        // }

        if (sigma > 1e-8f) {
            log_att = -world_step * sigma;
            const float weight = _EXP(log_light_intensity) * (1.f - _EXP(log_att));
            log_light_intensity += log_att;
            float* __restrict__ max_wt_ptr_000 = grid_weight.data() + idx;

            atomicMax(max_wt_ptr_000, weight);
            atomicMax(max_wt_ptr_000 + 1, weight);
            atomicMax(max_wt_ptr_000 + stride1, weight);
            atomicMax(max_wt_ptr_000 + stride1 + 1, weight);
            float* __restrict__ max_wt_ptr_100 = max_wt_ptr_000 + stride0;
            atomicMax(max_wt_ptr_100, weight);
            atomicMax(max_wt_ptr_100 + 1, weight);
            atomicMax(max_wt_ptr_100 + stride1, weight);
            atomicMax(max_wt_ptr_100 + stride1 + 1, weight);

            if (_EXP(log_light_intensity) < stop_thresh) {
                break;
            }
        }
        t += step_size;
    }
}

__launch_bounds__(MISC_CUDA_THREADS, MISC_MIN_BLOCKS_PER_SM)
    __global__ void grid_weight_render_kernel(
        const PackedVar32<float, 3>
            data,
        PackedCameraSpec cam,
        float step_size,
        float stop_thresh,
        bool last_sample_opaque,
        const float* __restrict__ offset,
        const float* __restrict__ scaling,
        PackedVar32<float, 3>
            grid_weight,
            float* test) {
    CUDA_GET_THREAD_ID(tid, cam.width() * cam.height());
    int iy = tid / cam.width(), ix = tid % cam.width();
    float dir[3], origin[3];
    cam2world_ray(ix, iy, cam, dir, origin);
//     if(tid==5535){
// printf("dir origin %f %f %f %f %f %f\n",dir[0],dir[1],dir[2],origin[0],origin[1],origin[2]);
// printf("aaaa %f %f\n",data(36437468),data(36437469));
// printf("bbbb %f %f\n",test[36437468],test[36437469]);
// printf("cccc %f %f\n",data.data()[36437468],data.data()[36437469]);
// printf("ptr %p %p\n",data.data(),test);
//     }
    grid_trace_ray(
        data,
        SingleRaySpec(origin, dir),
        offset,
        scaling,
        step_size,
        stop_thresh,
        last_sample_opaque,
        grid_weight,tid);
}