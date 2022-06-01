#include"ray_sampler_header.h"


template <typename TYPE>
__global__ void compacted_coord(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples_compacted,
	int padded_output_width,
	Array4f background_color,
	const TYPE *network_output,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	const NerfCoordinate *__restrict__ coords_in,
	NerfCoordinate *__restrict__ coords_out,
	const uint32_t *__restrict__ numsteps_in,
	uint32_t *__restrict__ numsteps_counter,
	uint32_t *__restrict__ numsteps_out,
	uint32_t *compacted_rays_counter)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_rays)
	{
		return;
	}

	uint32_t numsteps = numsteps_in[i * 2 + 0];
	uint32_t base = numsteps_in[i * 2 + 1];
	coords_in += base;
	network_output += base * 4;

	float T = 1.f;

	float EPSILON = 1e-4f;

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		if (T < EPSILON)
		{
			// break;
		}

		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		const Vector3f pos = unwarp_position(coords_in->pos.p, aabb);
		const float dt = unwarp_dt(coords_in->dt);

		float density = network_to_density(float(local_network_output[3]), density_activation);

		const float alpha = 1.f - __expf(-density * dt);

		T *= (1.f - alpha);
		network_output += 4;
		coords_in += 1;
	}

	network_output -= 4 * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_out[i * 2 + 0] = compacted_numsteps;
	numsteps_out[i * 2 + 1] = compacted_base;
	if (compacted_numsteps == 0)
	{
		return;
	}
	uint32_t rays_idx = atomicAdd(compacted_rays_counter, 1);
	coords_out += compacted_base;
	for (uint32_t j = 0; j < compacted_numsteps; ++j)
	{
		coords_out[j] = coords_in[j];
	}
}