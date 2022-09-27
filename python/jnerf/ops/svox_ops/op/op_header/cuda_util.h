#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define CUDA_GET_THREAD_ID(tid, Q)                         \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q)                                          \
    return
#define CUDA_GET_THREAD_ID_U64(tid, Q)                        \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q)                                             \
    return
#define CUDA_CHECK_ERRORS                 \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess)               \
    printf("Error in svox2.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))
__host__ __device__ __inline__ int getDeviceProperties() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        //TODO: repair device id
        return prop.maxThreadsPerBlock;
    }
    return 0;
}
#define CUDA_MAX_THREADS getDeviceProperties()

// Linear interp
// Subtract and fused multiply-add
// (1-w) a + w b
template <class T>
__host__ __device__ __inline__ T lerp(T a, T b, T w) {
    return fmaf(w, b - a, a);
}
__device__ __inline__ static float _rnorm(
    const float* __restrict__ dir) {
    // return 1.f / _norm(dir);
    return rnorm3df(dir[0], dir[1], dir[2]);
}
__host__ __device__ __inline__ static float _dot(
    const float* __restrict__ x,
    const float* __restrict__ y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

__device__ __inline__ void transform_coord(float* __restrict__ point,
                                           const float* __restrict__ scaling,
                                           const float* __restrict__ offset) {
    point[0] = fmaf(point[0], scaling[0], offset[0]);  // a*b + c
    point[1] = fmaf(point[1], scaling[1], offset[1]);  // a*b + c
    point[2] = fmaf(point[2], scaling[2], offset[2]);  // a*b + c
}
__device__ inline void atomicMax(float* result, float value) {
    unsigned* result_as_u = (unsigned*)result;
    unsigned old = *result_as_u, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_u, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (old != assumed);
    return;
}

__device__ inline void atomicMax(double* result, double value) {
    unsigned long long int* result_as_ull = (unsigned long long int*)result;
    unsigned long long int old = *result_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_ull, assumed,
                        __double_as_longlong(fmaxf(value, __longlong_as_double(assumed))));
    } while (old != assumed);
    return;
}

__device__ __inline__ static float _norm(
                const float* __restrict__ dir) {
    // return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    return norm3df(dir[0], dir[1], dir[2]);
}