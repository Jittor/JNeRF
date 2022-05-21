#include"density_grid_sampler_header.h"

template <typename T, typename T_OUT, typename F>
void reduce_sum(T *device_pointer, F fun, T_OUT *workspace, uint32_t n_elements, cudaStream_t stream, uint32_t n_sums = 1)
{
	const uint32_t threads = 1024;

	const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);

	if (n_elements % N_ELEMS_PER_LOAD != 0)
	{
		throw std::runtime_error{"Number of bytes to reduce_sum must be a multiple of 16."};
	}
	if (((size_t)device_pointer) % 16 != 0)
	{
		throw std::runtime_error{"Can only reduce_sum on 16-byte aligned memory."};
	}
	n_elements /= N_ELEMS_PER_LOAD;
	uint32_t blocks = div_round_up(n_elements, threads);
	block_reduce<T, T_OUT, F><<<blocks * n_sums, threads, 0, stream>>>(n_elements, fun, device_pointer, workspace, blocks);
}

__global__ void grid_to_bitfield(const uint32_t n_elements,
								 const float *__restrict__ grid,
								 uint8_t *__restrict__ grid_bitfield,
								 const float *__restrict__ mean_density_ptr)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;

	uint8_t bits = 0;

	float thresh = NERF_MIN_OPTICAL_THICKNESS() < *mean_density_ptr ? NERF_MIN_OPTICAL_THICKNESS() : *mean_density_ptr;

#pragma unroll
	for (uint8_t j = 0; j < 8; ++j)
	{
		bits |= grid[i * 8 + j] > thresh ? ((uint8_t)1 << j) : 0;
		
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(const uint32_t n_elements,
								  const uint8_t *__restrict__ prev_level,
								  uint8_t *__restrict__ next_level)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;

	uint8_t bits = 0;

#pragma unroll
	for (uint8_t j = 0; j < 8; ++j)
	{
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i * 8 + j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = morton3D_invert(i >> 0) + NERF_GRIDSIZE() / 8;
	uint32_t y = morton3D_invert(i >> 1) + NERF_GRIDSIZE() / 8;
	uint32_t z = morton3D_invert(i >> 2) + NERF_GRIDSIZE() / 8;

	next_level[morton3D(x, y, z)] |= bits;

}