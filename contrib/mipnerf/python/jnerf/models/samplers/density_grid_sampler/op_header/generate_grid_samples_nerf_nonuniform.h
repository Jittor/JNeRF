#include"density_grid_sampler_header.h"

__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements, default_rng_t rng, const uint32_t* step_p, BoundingBox aabb, const float *__restrict__ grid_in, NerfPosition *__restrict__ out, uint32_t *__restrict__ indices, uint32_t n_cascades, float thresh)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;
	// 1 random number to select the level, 3 to select the position.
	rng.advance(i * 4);
	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

	// Select grid cell that has density
	uint32_t idx;
	uint32_t step=*step_p;
	for (uint32_t j = 0; j < 10; ++j)
	{
		idx = ((i + step * n_elements) * 56924617 + j * 19349663 + 96925573) % (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());
		idx += level * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();
		if (grid_in[idx] > thresh)
		{
			break;
		}
	}

	// Random position within that cellq
	uint32_t pos_idx = idx % (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());

	uint32_t x = morton3D_invert(pos_idx >> 0);
	uint32_t y = morton3D_invert(pos_idx >> 1);
	uint32_t z = morton3D_invert(pos_idx >> 2);

	Eigen::Vector3f pos = ((Eigen::Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / NERF_GRIDSIZE() - Eigen::Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Eigen::Vector3f::Constant(0.5f);

	out[i] = {warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE())};
	indices[i] = idx;
}