#include <atomic>
#include <stdexcept>
#include "utils/log.h"
#include <stdio.h>
#include <cuda_fp16.h>
#define TCNN_HOST_DEVICE __host__ __device__
template <typename T>
TCNN_HOST_DEVICE T div_round_up(T val, T divisor)
{
	return (val + divisor - 1) / divisor;
}

template <typename T>
struct PitchedPtr
{
	TCNN_HOST_DEVICE PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
	TCNN_HOST_DEVICE PitchedPtr(T *ptr, size_t stride_in_elements, size_t offset = 0) : ptr{ptr + offset}, stride_in_bytes{(uint32_t)(stride_in_elements * sizeof(T))} {}

	template <typename U>
	TCNN_HOST_DEVICE explicit PitchedPtr(PitchedPtr<U> other) : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

	TCNN_HOST_DEVICE T *operator()(uint32_t y) const
	{
		return (T *)((const char *)ptr + y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE explicit operator bool() const
	{
		return ptr;
	}

	T *ptr;
	uint32_t stride_in_bytes;
};

template <typename T, uint32_t N_POS_DIMS>
__global__ void extract_position(
	const uint32_t num_elements,
	PitchedPtr<const float> data_in,
	T *__restrict__ output)
{

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements)
		return;

	const uint32_t dim_idx = threadIdx.y;

	output[i + dim_idx * num_elements] = (T)data_in(i)[dim_idx];
}

template <typename T, uint32_t N_ELEMS>
struct alignas(sizeof(T) * N_ELEMS) vector_t
{
	TCNN_HOST_DEVICE T &operator[](uint32_t idx)
	{
		return data[idx];
	}

	TCNN_HOST_DEVICE T operator[](uint32_t idx) const
	{
		return data[idx];
	}

	T data[N_ELEMS];
	static constexpr uint32_t N = N_ELEMS;
};
template <uint32_t N_DIMS>
__device__ uint32_t fast_hash(const uint32_t pos_grid[N_DIMS])
{
	static_assert(N_DIMS == 3, "fast_hash can only hash 3 dimensions.");
	return get_index(pos_grid[0], pos_grid[1], pos_grid[2]);
}
template <uint32_t N_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__device__ uint32_t grid_index(const uint32_t feature, const uint32_t hashmap_size, const uint32_t grid_resolution, const uint32_t pos_grid[N_DIMS])
{
	uint32_t stride = 1;
	uint32_t index = 0;

// The second part of the loop condition is needed to avoid integer overflows in finer levels.
#pragma unroll
	for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim)
	{
		index += pos_grid[dim] * stride;
		stride *= grid_resolution;
	}

	if (hashmap_size < stride)
	{
		index = fast_hash<N_DIMS>(pos_grid);
	}

	return (index % hashmap_size) * N_FEATURES_PER_LEVEL + feature;
}
__device__ inline float identity_fun(float val)
{
	return val;
}

__device__ inline float identity_derivative(float val)
{
	return 1;
}
template <uint32_t N_FLOATS>
using vector_fullp_t = vector_t<float, N_FLOATS>;
template <typename F, typename FPRIME>
__device__ inline void pos_fract(const float input, float *pos, float *pos_derivative, uint32_t *pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative)
{
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_grid(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const uint32_t *hashmap_offset_table,
	const uint32_t base_resolution,
	const float log2_per_level_scale,
	const float quantize_threshold,
	float max_level,
	const uint32_t interpolation_type,
	const uint32_t grid_type,
	const T *__restrict__ grid,
	const float *__restrict__ positions_in,
	vector_t<T, N_FEATURES_PER_LEVEL> *__restrict__ encoded_positions,
	float *__restrict__ dy_dx)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements)
		return;

	uint32_t level = blockIdx.y; // <- the level is the same for all threads.

	max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

	if (level >= max_level + 1e-3f)
	{
		encoded_positions[i + level * num_elements] = {};
		return;
	}

	grid += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];
	const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;

	const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);

	float pos[N_POS_DIMS];
	float pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
	{

		pos_fract(positions_in[i + dim * num_elements], &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, identity_fun, identity_derivative);
	}

	auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS])
	{
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(0, hashmap_size, grid_resolution, local_pos);
		return *(vector_t<T, N_FEATURES_PER_LEVEL> *)&grid[index];
	};

	// N-linear interpolation
	vector_t<T, N_FEATURES_PER_LEVEL> result = {};
#pragma unroll
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx)
	{
		float weight = 1;
		uint32_t pos_grid_local[N_POS_DIMS];
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
		{
			if ((idx & (1 << dim)) == 0)
			{
				weight *= 1 - pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			}
			else
			{
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		auto val = grid_val(pos_grid_local);
#pragma unroll
		for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature)
		{
			float data = (float)((T *)&val)[feature];
			if (fabsf(data) < quantize_threshold)
				data = 0.f;
			((T *)&result)[feature] += (T)(weight * data);
		}
	}

	encoded_positions[i + level * num_elements] = result;

	// Gradient
	if (dy_dx)
	{
#pragma unroll
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim)
		{
			vector_fullp_t<N_FEATURES_PER_LEVEL> grad = {0};

#pragma unroll
			for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx)
			{
				float weight = scale;
				uint32_t pos_grid_local[N_POS_DIMS];

#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim)
				{
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;

					if ((idx & (1 << non_grad_dim)) == 0)
					{
						weight *= 1 - pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					}
					else
					{
						weight *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				pos_grid_local[grad_dim] = pos_grid[grad_dim];
				auto val_left = grid_val(pos_grid_local);
				pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
				auto val_right = grid_val(pos_grid_local);

#pragma unroll
				for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature)
				{
					((float *)&grad)[feature] += weight * ((float)((T *)&val_right)[feature] - (float)((T *)&val_left)[feature]) * pos_derivative[grad_dim];
				}
			}

			const uint32_t fan_out_grad = num_grid_features * N_POS_DIMS;
			*(vector_fullp_t<N_FEATURES_PER_LEVEL> *)&dy_dx[i * fan_out_grad + level * N_FEATURES_PER_LEVEL + grad_dim * num_grid_features] = grad;
		}
	}
}

template <typename T>
__global__ void transpose_encoded_position(
	const uint32_t n_elements,
	const T *__restrict__ encoded_positions,
	PitchedPtr<T> output)
{
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements)
		return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	output(elem_idx)[dim_idx] = encoded_positions[elem_idx + n_elements * dim_idx];
}

template <typename T>
__global__ void transpose_gradients(
	const uint32_t n_elements,
	T *__restrict__ transposed_dL_dy,
	PitchedPtr<const T> dL_dy)
{
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements)
		return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	transposed_dL_dy[elem_idx + n_elements * dim_idx] = dL_dy(elem_idx)[dim_idx];
}
__device__ inline float smoothstep(float val)
{
	return val * val * (3.0f - 2.0f * val);
}

template <typename F>
__device__ inline void pos_fract(const float input, float *pos, uint32_t *pos_grid, float scale, F interpolation_fun)
{
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos = interpolation_fun(*pos);
}
template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_grid_backward(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const uint32_t *hashmap_offset_table,
	const uint32_t base_resolution,
	const float log2_per_level_scale,
	float max_level,
	const bool stochastic_interpolation,
	const uint32_t interpolation_type,
	const uint32_t grid_type,
	GRAD_T *__restrict__ grid_gradient,
	const float *__restrict__ positions_in,
	const vector_t<T, N_FEATURES_PER_THREAD> *__restrict__ dL_dy)
{
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements)
		return;

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads.
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

	if (level > max_level + 1e-3f)
	{
		printf("!!!! level return \n");
		return;
	}

	grid_gradient += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

	const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);

	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<T, N_FEATURES_PER_THREAD> &grad, const float weight)
	{
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(feature, hashmap_size, grid_resolution, local_pos);
		// // #if TCNN_MIN_GPU_ARCH >= 60 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEATURES_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value)
		{
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; f += 2)
			{

				__half2 v = {(__half)((float)grad[f] * weight), (__half)((float)grad[f + 1] * weight)};
				atomicAdd((__half2 *)&grid_gradient[index + f], v);
			}
		}
		else
		// #endif
		{
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f)
			{
				atomicAdd(&grid_gradient[index + f], (T)((float)grad[f] * weight));
			}
		}
	};

	float pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	{
#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
		{
			pos_fract(positions_in[i + dim * num_elements], &pos[dim], &pos_grid[dim], scale, identity_fun);
		}
	}

	auto grad = dL_dy[(i + level * num_elements) * (N_FEATURES_PER_LEVEL) / N_FEATURES_PER_THREAD + feature / N_FEATURES_PER_THREAD];

	// N-linear interpolation
#pragma unroll
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx)
	{
		float weight = 1;
		uint32_t pos_grid_local[N_POS_DIMS];
#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
		{
			if ((idx & (1 << dim)) == 0)
			{
				weight *= 1 - pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			}
			else
			{
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		add_grid_gradient(pos_grid_local, grad, weight);
	}
}
