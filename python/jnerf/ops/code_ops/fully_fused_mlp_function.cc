#pragma once
#undef out
#include "fully_fused_mlp_header.h"
// implement temp GPUMatrix and GPUDynamicMatrix here.
#define RM MatrixLayout::kRowMajor
#define CM MatrixLayout::kColumnMajor

template <typename T>
__device__ __host__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}
// warp activation defined here
template <typename T, typename fragment_t>
__host__ __device__ void warp_activation(Activation activation, const fragment_t& frag, fragment_t& result) {
    switch (activation) {
        case Activation::ReLU:
            #pragma unroll
            for (int t=0; t < result.num_elements; t++) {
                result.x[t] = frag.x[t] * (T)((T)frag.x[t] > (T)0.0f);
            }
            return;
        case Activation::None: result = frag; return;
        default:
            // Unsupported activation
            // assert(false); // Commented out due to isolated strange side-effects on Windows
            return;
    }
}
template <typename T, typename fragment_t>
__host__ __device__ fragment_t warp_activation(Activation activation, const fragment_t& frag) {
    fragment_t result;
    warp_activation<T>(activation, frag, result);
    return result;
}
template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ void warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag, fragment_t& result) {
    switch (activation) {
        case Activation::ReLU:
            #pragma unroll
            for (int t=0; t < result.num_elements; t++) {
                result.x[t] = frag.x[t] * (T)(forward_frag.x[t] > (T)0.0f);
            }
            return;
        case Activation::None: result = frag; return;
        default:
            // Unsupported activation
            // assert(false); // Commented out due to isolated strange side-effects on Windows
            return;
    }
}
template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ fragment_t warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag) {
    fragment_t result;
    warp_activation_backward<T>(activation, frag, forward_frag, result);
    return result;
}
void check_shmem_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` or use `CutlassMLP` (better compatibility but slower) instead."};
    }
}
template <int WIDTH, int N_ITERS, typename OUT_T, bool BACKWARD=false>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr) {
    // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
    //           Can be forward activations or backward activations, depending on caller.
    // weights_this_layer points to the weight matrix of the current layer.
    // out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
    //                  Can be nullptr if nothing should be written.
    // activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.
    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    constexpr uint32_t N_BLOCKS = WIDTH / 16;
    using namespace nvcuda;
    // If we're performing the backward pass, weights must be loaded in transposed form, which
    // is achieved by interpreting the memory in row_major instead of col_major order.
    using weights_layout_t = std::conditional_t<BACKWARD, wmma::row_major, wmma::col_major>;
    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, weights_layout_t> weights_frag[N_BLOCKS];
    wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")
    const uint32_t lane_offset = (8 * li) % WIDTH;
    const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;
    const uint32_t weights_col = 16 * wi;
    __syncthreads();
    // Load N_BLOCKS chunks of weights from global memory into registers.
    #pragma unroll
    for (uint32_t i = 0; i < N_BLOCKS; ++i) {
        if (BACKWARD) {
            // If we're performing the backward pass, additional index swizzling is needed to
            // load the weights in transposed form.
            wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i * WIDTH + weights_col, WIDTH);
        } else {
            wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH);
        }
    }
    #pragma unroll
    for (int l = 0; l < N_ITERS; ++l) {
        wmma::fill_fragment(result_frag[l], 0.0f);
        #pragma unroll
        for (uint32_t i = 0; i < N_BLOCKS; ++i) {
            // Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
            wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * l) * (WIDTH + SKEW), WIDTH + SKEW);
            wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
        }
        // Activation
        if (BACKWARD) {
            // Load the temporary forward matrix for the relu transfer
            wmma::load_matrix_sync(act_frag, activation_aux + weights_col + l * 16 * WIDTH, WIDTH);
            warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
        } else {
            warp_activation<__half>(activation, result_frag[l], result_frag[l]);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int l = 0; l < N_ITERS; ++l) {
        wmma::store_matrix_sync(act_shmem + weights_col + l * 16 * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
    }
    if (out_intermediate_threadblock_this_layer != nullptr) {
        __syncthreads();
        #pragma unroll
        for (int l = 0; l < N_ITERS; ++l) {
            *(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * l) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * l) * (WIDTH + SKEW)];
        }
    }
}
template <int WIDTH, int N_ITERS>
__device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) {
    // act_shmem will be filled by the thread block's chunk of input_threadblock
    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")
    const uint32_t lane_offset = (8 * li) % WIDTH;
    const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;
    #pragma unroll
    for (int i = 0; i < N_ITERS; ++i) {
        *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * i) * WIDTH];
    }
}
template <int WIDTH, int N_ITERS, Activation ACTIVATION, typename OUTPUT_LAYOUT>
__global__ void kernel_mlp_fused_backward(const __half* __restrict__ dL_doutput, const __half* __restrict__ weights, __half* __restrict__ out_intermediate, const __half* __restrict__ forward, __half* __restrict__ dL_dinput, const __half* __restrict__ weights_first_layer, const uint32_t batch_size, const uint32_t out_width, const uint32_t n_hidden_matmuls, int need_last) {
    // `dL_doutput` points to the input matrix of the backward pass, i.e. the loss gradients. Assumed to be 16 neurons wide.
    // `weights` points to the weight matrices (contiguous in memory).
    // `out_intermediate` points to the memory where backpropagated activation gradients should be written.
    // `forward` points to the memory where the intermediate activations of the forward pass are located. (needed for activation backprop)
    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")
    const uint32_t bi = blockIdx.x;	 // block index
    // Shared memory contains the intermediate activations of blockDim.y*16 elements.
    // A skew is applied to the matrix storage to avoid bank conflicts.
    extern __shared__ __half shmem[];
    __half* act_shmem = shmem;
    const uint32_t lane_offset = (8 * li) % WIDTH;
    const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;
    // Multipying one 16-row chunk of intermediate activations with the weight matrix requires all warps of the block.
    // Thus, each block computes exactly one 16-row chunk of the next layer's intermediate activations.
    const uint32_t elem_idx_base = 16 * bi * N_ITERS;
    const uint32_t elem_idx = elem_idx_base;
    const uint32_t layer_stride = WIDTH * WIDTH;
    const uint32_t output_stride = WIDTH * batch_size;
    // Backprop through last layer
    if (out_width <= 16) {
        using namespace nvcuda;
        // Fragments in registers
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, OUTPUT_LAYOUT> act_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> weights_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];
        // Load the relevant chunk of the last layer's weight matrix from global memory into registers
        const uint32_t weights_col = 16 * wi;
        wmma::load_matrix_sync(weights_frag, weights + layer_stride * n_hidden_matmuls + weights_col, WIDTH);
        #pragma unroll
        for (int l = 0; l < N_ITERS; ++l) {
            wmma::fill_fragment(result_frag[l], 0.0f);
            // Load a chunk of output gradients from shared memory and multiply with previously loaded weights
            if (std::is_same<OUTPUT_LAYOUT, wmma::row_major>::value) {
                wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * l) * 16, 16);
            } else {
                wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * l), batch_size);
            }
            // NOTE: activation transfer of the _output_ activation is expected to be done _prior_ to calling this kernel
            //       in a separate pass, because the tranfered activation gradient is also needed to compute the weight
            //       gradient of the last weight matrix (see backward()).
            wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
            // Load the temporary forward matrix for the relu transfer
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> forward_frag;
            wmma::load_matrix_sync(forward_frag, forward + output_stride * n_hidden_matmuls + weights_col + (elem_idx + l * 16) * WIDTH, WIDTH);
            warp_activation_backward<__half>(ACTIVATION, result_frag[l], forward_frag, result_frag[l]);
        }
        __syncthreads();
        #pragma unroll
        for (int l = 0; l < N_ITERS; ++l) {
            wmma::store_matrix_sync(act_shmem + weights_col + (16 * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < N_ITERS; ++i) {
            *(int4*)&out_intermediate[lane_offset + (row + elem_idx + i * 16) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
        }
    } 
    // else {
    // 	// If the output width is larger than 16, we will have used CUTLASS for backpropping through the last layer.
    // 	// Load the resulting gradients.
    // 	threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
    // }
    // Backprop through hidden layers
    for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
        threadblock_layer<WIDTH, N_ITERS, __half, true>(ACTIVATION, act_shmem, weights + layer_stride * (n_hidden_matmuls - k - 1), out_intermediate + output_stride * (k + 1) + elem_idx_base * WIDTH, forward + output_stride * (n_hidden_matmuls - k - 1) + elem_idx_base * WIDTH);
    }
    // Compute loss gradients w.r.t. input if desired.
    // THIS CODE ASSUMES THAT THE INPUT WIDTH IS THE SAME AS THE NETWORK WIDTH
    // AND THAT THE INPUT LAYOUT IS THE SAME AS THE HIDDEN LAYOUT.
    // DON'T PASS A NON-NULL dL_dinput IF THIS REQUIREMENT IS NOT MET.
    if (dL_dinput != nullptr && need_last) {
        threadblock_layer<WIDTH, N_ITERS, __half, true>(Activation::None, act_shmem, weights_first_layer, dL_dinput + elem_idx_base * WIDTH);
    }
}
template <int WIDTH, typename T, Activation ACTIVATION>
void mlp_fused_backward(
    cudaStream_t stream,
    T* weights_first_layer,
    T* weights,
    T* dL_doutput,
    T* temps,
    T* forward,
    T* dL_dinput,
    const uint32_t n_hidden_matmuls,
    int grad_shape0,
    int grad_shape1,
    int need_last
) {
    const uint32_t batch_size = grad_shape0;
    const uint32_t out_width = grad_shape1;
    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    constexpr uint32_t N_BLOCKS = WIDTH / 16;
    // if (forward.cols() != batch_size) {
    // 	throw std::runtime_error{"Batch size of matrices dL_doutput and temporaries doesn't match."};
    // }
    const int N_ITERS = WIDTH >= 256 ? 2 : 8;
    // if (batch_size % (16 * N_ITERS) != 0) {
    // 	throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS) + "."};
    // }
    const dim3 threads = { 32u, N_BLOCKS, 1 }; // 32 threads = 1 warp, 8 warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)
    uint32_t n_elems_per_block = 16 * N_ITERS;
    uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);
    int shmem_size = sizeof(__half) * ((16 * N_ITERS) * (WIDTH + SKEW)); // WIDTH rows of input and 16 * threads.z rows of weights
    const dim3 blocks = { n_blocks, 1u, 1u };
    // The kernels operate with transposed layouts compared with the MLP code
    // if (dL_doutput.layout() == RM) {
    // 	check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
    // 	kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
    // } else {
    // 	check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
    // 	kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::row_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
    // }
    check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
    kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput, weights, temps, forward, dL_dinput, weights_first_layer, batch_size, out_width, n_hidden_matmuls, need_last);
}
template <int WIDTH, int N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width, const uint32_t batch_size) {
    // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
    // input_threadblock points to the thread block's chunk of the input batch in global memory
    // weights_this_layer points to the weight matrix of the current layer
    // out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
    //                  Can be nullptr if nothing should be written.
    // in_width is the dynamic width of the input layer
    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    constexpr uint32_t INPUT_SKEW = 8;
    constexpr uint32_t N_BLOCKS = WIDTH / 16;
    using namespace nvcuda;
    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, INPUT_LAYOUT> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")
    const uint32_t lane_offset = (8 * li) % WIDTH;
    const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;
    const uint32_t weights_col = 16 * wi;
    __half* __restrict__ weights_shmem = act_shmem + 16 * (in_width + INPUT_SKEW);
    // Load input weight matrix (fits completely into shared memory)
    // Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS warps
    const uint32_t n_elems_per_load = N_BLOCKS * 32 * 8;
    const uint32_t thread_elem_idx = (li + wi * 32) * 8;
    const uint32_t n_elems_b = WIDTH * in_width;
    #pragma unroll
    for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
        const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
        *(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
    }
    const uint32_t n_tensor_ops = in_width / 16;
    if (std::is_same<INPUT_LAYOUT, wmma::col_major>::value) {
        __syncthreads();
    }
    #pragma unroll
    for (int l = 0; l < N_ITERS; ++l) {
        if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
            // Load chunk of inputs into shmem.
            // This is faster than loading it from gmem directly, even though it is only used once.
            // (Possibly due to latency hiding through staging.)
            const uint32_t n_elems_a = 16 * in_width;
            #pragma unroll
            for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
                const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
                *(int4*)&act_shmem[idx_skewed] = *(int4*)&input_threadblock[l * n_elems_a + idx];
            }
            __syncthreads();
        }
        wmma::fill_fragment(result_frag[l], 0.0f);
        #pragma unroll
        for (uint32_t i = 0; i < n_tensor_ops; ++i) {
            // Load chunk of inputs and weights from shared memory and multiply them
            if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
                wmma::load_matrix_sync(act_frag, act_shmem + 16 * i, in_width + INPUT_SKEW);
            } else {
                wmma::load_matrix_sync(act_frag, input_threadblock + 16 * i * batch_size + 16 * l, batch_size);
            }
            wmma::load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
            wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
        }
        if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
            __syncthreads();
        }
        warp_activation<__half>(activation, result_frag[l], result_frag[l]);
    }
    if (std::is_same<INPUT_LAYOUT, wmma::col_major>::value) {
        __syncthreads();
    }
    #pragma unroll
    for (int l = 0; l < N_ITERS; ++l) {
        wmma::store_matrix_sync(act_shmem + weights_col + (16 * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
    }
    if (out_intermediate_threadblock_this_layer != nullptr) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < N_ITERS; ++i) {
            *(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
        }
    }
}

template <int WIDTH, int N_ITERS, typename OUT_T>
__device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const uint32_t batch_size, const nvcuda::wmma::layout_t output_layout, const uint32_t output_stride) {
    // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
    // weights_this_layer points to the weight matrix of the current layer
    // out points to the location where the result produced by the thread block should be written to.
    //   Can be nullptr if nothing should be written.
    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    constexpr uint32_t N_BLOCKS = WIDTH / 16;
    using namespace nvcuda;
    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
    wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag;
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")
    __half* __restrict__ weights_shmem = act_shmem + N_ITERS * 16 * (WIDTH + SKEW);
    const uint32_t weights_row = (8 * li) % WIDTH;
    const uint32_t weights_col = (8 * li + 8 * 32 * wi) / WIDTH;
    // Load weight matrix into shared memory for the last multiplication.
    // Loading into shared memory as opposed to directly into registers is faster
    // because unlike in the previous layers, each warp uses the same entries of the weight matrix.
    *(int4*)&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];
    __syncthreads();
    #pragma unroll
    for (uint32_t i = 0; i < N_BLOCKS; ++i)
        wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16 * i, WIDTH + SKEW);
    // Perform last layer by parallelizing over iters
    for (uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
        wmma::fill_fragment(result_frag, 0.0f);
        #pragma unroll
        for (uint32_t i = 0; i < N_BLOCKS; ++i) {
            // Load a chunk of intermediate activations from shared memory and multiply with chunk of the weight matrix
            wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * idx) * (WIDTH + SKEW), WIDTH + SKEW);
            wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
        }

        warp_activation<__half>(activation, result_frag, result_frag);
        if (output_layout == wmma::mem_row_major) {
            wmma::store_matrix_sync(out + idx * 16 * output_stride, result_frag, output_stride, output_layout);
        } else {
            wmma::store_matrix_sync(out + idx * 16, result_frag, batch_size, output_layout);
        }
    }
}
template <int WIDTH, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
__global__ void kernel_mlp_fused(const Activation output_activation, const __half* __restrict__ input, const __half* __restrict__ weights, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const uint32_t batch_size, const uint32_t in_width, const uint32_t out_width, const uint32_t n_hidden_matmuls, const nvcuda::wmma::layout_t input_layout, const nvcuda::wmma::layout_t output_layout, const int output_stride) {
    // `input` points to the input matrix. Can be any width.
    // `weights` points to the weight matrices (contiguous in memory).
    // `out_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
    // `out` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)
    // Commented out due to isolated strange side-effects on Windows
    // if (INFERENCE) {
    // 	assert(out_intermediate == nullptr);
    // } else {
    // 	assert(out_intermediate);
    // }
    // Shared memory contains the intermediate activations of blockDim.y*16 elements.
    // In some cases, it also contains the weight matrix for the first and last layer.
    extern __shared__ __half shmem[];
    __half* act_shmem = shmem;
    // Each block computes exactly one 16-element chunk of the batch.
    const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS;
    // First layer
    if (input_layout == nvcuda::wmma::mem_col_major || in_width != WIDTH) {
        if (input_layout == nvcuda::wmma::mem_row_major) {
            threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
        } else {
            threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
        }
    } else {
        // If the input has the same width & layout as the hidden layers, we can simply use the network's regular layer routine (with static size)
        // instead of using the slower dynamic input layer routine.
        threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
        threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
    }
    const uint32_t first_layer_size = WIDTH * in_width;
    const uint32_t weights_stride = WIDTH * WIDTH;
    const uint32_t layer_stride = WIDTH * batch_size;
    // Hidden layers
    for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
        threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + weights_stride * k, !INFERENCE ? (out_intermediate + layer_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
    }
    // Last layer
    if (output_layout == nvcuda::wmma::mem_row_major) {                
        threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + weights_stride * n_hidden_matmuls, out + elem_idx * output_stride, output_stride, output_layout, output_stride);
    } else {
        threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + weights_stride * n_hidden_matmuls, out + elem_idx, batch_size, output_layout, output_stride);
    }
    
}
template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
void mlp_fused_forward(
    cudaStream_t stream,
    Activation output_activation,
    T* weights,
    T* input,
    T* output_intermediate,
    T* output,
    const uint32_t n_hidden_layers,
    int input_shape0,
    int input_shape1,
    int weights_shape0,
    int weights_shape1,
    int output_shape0,
    int output_shape1
) {
    const uint32_t batch_size = input_shape0;
    const uint32_t in_width = input_shape1;
    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
    constexpr uint32_t INPUT_SKEW = 8; // <- likewise with inputs
    constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;
    // LOGir << batch_size << " " << in_width << " " << WIDTH << " " << weights_shape0 << " " <<weights_shape1;
    // static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
    // if (in_width % 16 != 0) {
    // 	throw std::runtime_error{"Inputs must have a multiple-of-16 elements."};
    // }
    // if (weights.rows() != WIDTH) {
    // 	throw std::runtime_error{"The fully fused forward pass only works with WIDTH-sized matrices."};
    // }
    // if (weights.cols() % 16 != 0) {
    // 	throw std::runtime_error{std::string("weights must have a multiple-of-16 number of columns. ") + std::to_string(weights.cols())};
    // }
    // if (output_intermediate.cols() != batch_size) {
    // 	throw std::runtime_error{"Batch size of inputs and output_intermediate doesn't match."};
    // }
    // if (output && output->cols() != batch_size) {
    // 	throw std::runtime_error{"Batch size of inputs and outputs doesn't match."};
    // }
    const int N_ITERS = WIDTH >= 256 ? 2 : 8;
    if (batch_size % (16 * N_ITERS) != 0) {
        throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS) + "."};
    }
    const dim3 threads = { 32u, N_BLOCK_ROWS, 1 }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)
    uint32_t n_elems_per_block = 16 * N_ITERS;
    uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);
    size_t shmem_size = sizeof(__half) * (16 + 16 * N_ITERS) * (WIDTH + SKEW); // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*N_ITERS rows of intermediate activations
    shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16) * (in_width + INPUT_SKEW));
    const dim3 blocks = { n_blocks, 1u, 1u };
    check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
    kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE><<<blocks, threads, shmem_size, stream>>>(
        output_activation,
        input,
        weights,
        output_intermediate,
        output ? output : nullptr,
        batch_size,
        in_width,
        output ? output_shape1 : 0,
        n_hidden_layers,
        // The kernels operate with transposed layouts compared with the MLP code
        nvcuda::wmma::mem_row_major,
        nvcuda::wmma::mem_row_major,
        output_shape1
    );
}


void mlp_fused_forward_func(
    int WIDTH,
    Activation ACTIVATION,
    bool INFERENCE,
    cudaStream_t stream,
    Activation output_activation,
    __half* weights,
    __half* input,
    __half* output_intermediate,
    __half* output,
    const uint32_t n_hidden_layers,
    int input_shape0,
    int input_shape1,
    int weights_shape0,
    int weights_shape1,
    int output_shape0,
    int output_shape1
) {

    #define FORWARD_ARGS \
     stream,\
     output_activation,\
     weights,\
     input,\
     output_intermediate,\
     output,\
     n_hidden_layers,\
     input_shape0,\
     input_shape1,\
     weights_shape0,\
     weights_shape1,\
     output_shape0,\
     output_shape1
    if (WIDTH == 64 && ACTIVATION==Activation::ReLU && INFERENCE==false) {
        mlp_fused_forward<64, __half, Activation::ReLU, false>(FORWARD_ARGS);
    } 
    else {
        std::string msg = "mlp_fused_forward_func error not supported WIDTH=" + std::string("WIDTH") + " ACTIVATION=" + std::string("ACTIVATION") + " INFERENCE=" + std::string("INFERENCE") + ", please contact us to add this support.";
    }
}


void mlp_fused_backward_func(
    int WIDTH, 
    Activation ACTIVATION,
    cudaStream_t stream,
    __half* weights_first_layer,
    __half* weights,
    __half* dL_doutput,
    __half* temps,
    __half* forward,
    __half* dL_dinput,
    const uint32_t n_hidden_matmuls,
    int grad_shape0,
    int grad_shape1,
    int need_last
) {
    #define BACKWARD_ARGS \
    stream, \
    weights_first_layer, \
    weights, \
    dL_doutput, \
    temps, \
    forward, \
    dL_dinput, \
    n_hidden_matmuls, \
    grad_shape0, \
    grad_shape1, \
    need_last
    if (WIDTH == 64 && ACTIVATION==Activation::ReLU) {
        mlp_fused_backward<64, __half, Activation::ReLU>(BACKWARD_ARGS);
    } 
    else {
        std::string msg = "mlp_fused_forward_func error not supported WIDTH=" + std::string("WIDTH") + " ACTIVATION=" + std::string("ACTIVATION") + " INFERENCE=" + std::string("INFERENCE") + ", please contact us to add this support.";
    }
}