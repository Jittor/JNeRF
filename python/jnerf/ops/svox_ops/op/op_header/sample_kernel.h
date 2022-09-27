

#include <stdio.h>
#include "cuda_util.h"
#include "data_spec.h"

#include <cstdint>
#include "jt_helper.h"
#include "render_util.cuh"
__global__ void sample_grid_sh_kernel(
    PackedSparseGridSpec grid,
    const PackedVar32<float, 2> points,
    PackedVar32<float, 2> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0) * grid.sh_data_dim);
    const int idx = tid % grid.sh_data_dim;
    const int pid = tid / grid.sh_data_dim;

    float point[3] = {points[pid][0], points[pid][1], points[pid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], (int32_t)(grid.size[i] - 2));
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? grid.sh_data[link_ptr[u] * size_t(grid.sh_data_dim) + idx] : 0.f)

    const float ix0y0 = lerp(MAYBE_READ_LINK(0), MAYBE_READ_LINK(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK(offy), MAYBE_READ_LINK(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK(offx), MAYBE_READ_LINK(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK(offy + offx),
                             MAYBE_READ_LINK(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[pid][idx] = lerp(ix0, ix1, point[0]);
}
#undef MAYBE_READ_LINK

__global__ void sample_grid_density_kernel(
    PackedSparseGridSpec grid,
    const PackedVar32<float, 2> points,
    PackedVar32<float, 2> out) {
    CUDA_GET_THREAD_ID(tid, points.size(0));

    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

#define MAYBE_READ_LINK_D(u) ((link_ptr[u] >= 0) ? grid.density_data[link_ptr[u]] : 0.f)

    const float ix0y0 = lerp(MAYBE_READ_LINK_D(0), MAYBE_READ_LINK_D(1), point[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK_D(offy), MAYBE_READ_LINK_D(offy + 1), point[2]);
    const float ix0 = lerp(ix0y0, ix0y1, point[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK_D(offx), MAYBE_READ_LINK_D(offx + 1), point[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK_D(offy + offx),
                             MAYBE_READ_LINK_D(offy + offx + 1), point[2]);
    const float ix1 = lerp(ix1y0, ix1y1, point[1]);
    out[tid][0] = lerp(ix0, ix1, point[0]);
}
#undef MAYBE_READ_LINK_D

__global__ void sample_grid_sh_backward_kernel(
    PackedSparseGridSpec grid,
    const PackedVar32<float, 2> points,
    const PackedVar32<float, 2> grad_out,
    // output
    PackedVar64<float, 2> grad_data) {
    CUDA_GET_THREAD_ID(tid, points.size(0) * grid.sh_data_dim);
    const int idx = tid % grid.sh_data_dim;
    const int pid = tid / grid.sh_data_dim;

    float point[3] = {points[pid][0], points[pid][1], points[pid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

    const float go = grad_out[pid][idx];

    const float xb = point[0], yb = point[1], zb = point[2];
    const float xa = 1.f - point[0], ya = 1.f - point[1], za = 1.f - point[2];

#define MAYBE_ADD_GRAD_LINK_PTR(u, content) \
    if (link_ptr[u] >= 0)                   \
    atomicAdd(&grad_data[link_ptr[u]][idx], content)

    const float xago = xa * go;
    float tmp = ya * xago;
    MAYBE_ADD_GRAD_LINK_PTR(0, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(1, tmp * zb);
    tmp = yb * xago;
    MAYBE_ADD_GRAD_LINK_PTR(offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(offy + 1, tmp * zb);

    const float xbgo = xb * go;
    tmp = ya * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR(offx, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(offx + 1, tmp * zb);
    tmp = yb * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR(offx + offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR(offx + offy + 1, tmp * zb);
}
#undef MAYBE_ADD_GRAD_LINK_PTR

__global__ void sample_grid_density_backward_kernel(
    PackedSparseGridSpec grid,
    const PackedVar32<float, 2> points,
    const PackedVar32<float, 2> grad_out,
    // output
    PackedVar32<float, 2> grad_data) {
    CUDA_GET_THREAD_ID(tid, points.size(0));

    float point[3] = {points[tid][0], points[tid][1], points[tid][2]};
    transform_coord(point, grid._scaling, grid._offset);

    int32_t l[3];
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        point[i] = fminf(fmaxf(point[i], 0.f), grid.size[i] - 1.f);
        l[i] = min((int32_t)point[i], grid.size[i] - 2);
        point[i] -= l[i];
    }

    const int offy = grid.size[2], offx = grid.size[1] * grid.size[2];
    const int32_t* __restrict__ link_ptr = &grid.links[l[0] * offx + l[1] * offy + l[2]];

    const float go = grad_out[tid][0];

    const float xb = point[0], yb = point[1], zb = point[2];
    const float xa = 1.f - point[0], ya = 1.f - point[1], za = 1.f - point[2];

#define MAYBE_ADD_GRAD_LINK_PTR_D(u, content) \
    if (link_ptr[u] >= 0)                     \
    atomicAdd(grad_data[link_ptr[u]].data(), content)

    const float xago = xa * go;
    float tmp = ya * xago;
    MAYBE_ADD_GRAD_LINK_PTR_D(0, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(1, tmp * zb);
    tmp = yb * xago;
    MAYBE_ADD_GRAD_LINK_PTR_D(offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(offy + 1, tmp * zb);

    const float xbgo = xb * go;
    tmp = ya * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR_D(offx, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(offx + 1, tmp * zb);
    tmp = yb * xbgo;
    MAYBE_ADD_GRAD_LINK_PTR_D(offx + offy, tmp * za);
    MAYBE_ADD_GRAD_LINK_PTR_D(offx + offy + 1, tmp * zb);
}