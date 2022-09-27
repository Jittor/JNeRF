#pragma once
#include "jt_helper.h"
#include "cuda_util.h"

const int WARP_SIZE = 32;
const int TV_GRAD_CUDA_THREADS = 256;
const int TV_GRAD_POINTS_PER_BLOCK = TV_GRAD_CUDA_THREADS / WARP_SIZE;
const int MIN_BLOCKS_PER_SM = 4;

__device__ __inline__
void calculate_ray_scale(
                         float z,
                         float maxx,
                         float maxy,
                         float maxz,
                         float* __restrict__ scale) {
    // if (ndc_coeffx > 0.f) {
    //     // FF NDC
    //     scale[0] = maxx * (1.f / 256.f);
    //     scale[1] = maxy * (1.f / 256.f);
    //     scale[2] = maxz * (1.f / 256.f);

        // The following shit does not work
        // // Normalized to [-1, 1] (with 0.5 padding)
        // // const float x_norm = (x + 0.5) / maxx * 2 - 1;
        // // const float y_norm = (y + 0.5) / maxy * 2 - 1;
        // const float z_norm = (z + 0.5) / maxz * 2 - 1;
        //
        // // NDC distances
        // const float disparity = (1 - z_norm) / 2.f; // in [0, 1]
        // scale[0] = (ndc_coeffx * disparity);
        // scale[1] = (ndc_coeffy * disparity);
        // scale[2] = -((z_norm - 1.f + 2.f / maxz) * disparity) / (maxz * 0.5f);
    // } else {
        scale[0] = maxx * (1.f / 256.f);
        scale[1] = maxy * (1.f / 256.f);
        scale[2] = maxz * (1.f / 256.f);
    // }
}


#define CALCULATE_RAY_SCALE(out_name, maxx, maxy, maxz) \
    calculate_ray_scale( \
            z, \
            maxx, \
            maxy, \
            maxz, \
            out_name)

// __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void tv_grad_sparse_kernel(
    const PackedVar32<int32_t,3>links,
    const PackedVar64<float,2>data,
        const int32_t* __restrict__ rand_cells,
        int start_dim, int end_dim,
        float scale,
        size_t Q,
        bool ignore_edge,
        bool ignore_last_z,
        // float ndc_coeffx, float ndc_coeffy,
        // Output
        bool* __restrict__ mask_out,
        float* __restrict__ grad_data) {
    CUDA_GET_THREAD_ID_U64(tid, Q);
    const int idx = tid % (end_dim - start_dim) + start_dim;
    const int xyz = rand_cells[tid / (end_dim - start_dim)];
    const int z = xyz % links.size(2);
    const int xy = xyz / links.size(2);
    const int y = xy % links.size(1);
    const int x = xy / links.size(1);

    const int32_t* __restrict__ links_ptr = &links[x][y][z];

    if (ignore_edge && *links_ptr == 0) return;

    float scaling[3];
    CALCULATE_RAY_SCALE(scaling, links.size(0), links.size(1), links.size(2));

    const int offx = links.stride(0), offy = links.stride(1);

    const auto lnk000 = links_ptr[0];
    const auto lnk001 = ((z + 1 < links.size(2)) &&
                         (!ignore_last_z || z != links.size(2) - 2)) ?
                        links_ptr[1] : 0;
    const auto lnk010 = y + 1 < links.size(1) ? links_ptr[offy] : 0;
    const auto lnk100 = x + 1 < links.size(0) ? links_ptr[offx] : 0;
    if (ignore_last_z && z == links.size(2) - 2) return;

    const float v000 = lnk000 >= 0 ? data[lnk000][idx] : 0.f;
    const float null_val = (ignore_edge ? v000 : 0.f);
    const float v001 = lnk001 >= 0 ? data[lnk001][idx] : null_val,
                v010 = lnk010 >= 0 ? data[lnk010][idx] : null_val,
                v100 = lnk100 >= 0 ? data[lnk100][idx] : null_val;

    float dx = (v100 - v000);
    float dy = (v010 - v000);
    float dz = (v001 - v000);
    const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);

    dx *= scaling[0];
    dy *= scaling[1];
    dz *= scaling[2];

#define MAYBE_ADD_SET(lnk, val) if (lnk >= 0 && val != 0.f) { \
    atomicAdd(&grad_data[lnk * data.size(1) + idx], val * idelta); \
    if (mask_out != nullptr) { \
        mask_out[lnk] = true; \
    } \
} \

    const float sm = -(dx + dy + dz);
    MAYBE_ADD_SET(lnk000, sm);
    MAYBE_ADD_SET(lnk001, dz);
    MAYBE_ADD_SET(lnk010, dy);
    MAYBE_ADD_SET(lnk100, dx);

#undef MAYBE_ADD_SET
}