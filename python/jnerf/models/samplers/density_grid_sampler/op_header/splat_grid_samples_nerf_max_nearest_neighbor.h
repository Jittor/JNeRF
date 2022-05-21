#include"density_grid_sampler_header.h"


template <typename T>
__global__ void splat_grid_samples_nerf_max_nearest_neighbor(const uint32_t n_elements, const uint32_t *__restrict__ indices, int padded_output_width, const T *network_output, float *__restrict__ grid_out, ENerfActivation rgb_activation, ENerfActivation density_activation)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;

	uint32_t local_idx = indices[i];

	// Current setting: optical thickness of the smallest possible stepsize.
	// Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
	uint32_t level = 0; // local_idx / (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());

	float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
	float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);

	// Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
	// uint atomicMax is thus perfectly acceptable.
	atomicMax((uint32_t *)&grid_out[local_idx], __float_as_uint(optical_thickness));
}
