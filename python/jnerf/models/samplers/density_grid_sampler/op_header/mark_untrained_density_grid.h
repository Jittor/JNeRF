#include"density_grid_sampler_header.h"

__global__ void mark_untrained_density_grid(const uint32_t n_elements, float *__restrict__ grid_out,
											const uint32_t n_training_images,
											const Vector2f *__restrict__ focal_lengths,
											const Matrix<float, 3, 4> *training_xforms,
											Vector2i resolution)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;
	uint32_t level = i / (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());
	uint32_t pos_idx = i % (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());

	uint32_t x = morton3D_invert(pos_idx >> 0);
	uint32_t y = morton3D_invert(pos_idx >> 1);
	uint32_t z = morton3D_invert(pos_idx >> 2);

	float half_resx = resolution.x() * 0.5f;
	float half_resy = resolution.y() * 0.5f;

	Vector3f pos = ((Vector3f{(float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f}) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);
	float voxel_radius = 0.5f * SQRT3() * scalbnf(1.0f, level) / NERF_GRIDSIZE();
	int count = 0;
	for (uint32_t j = 0; j < n_training_images; ++j)
	{
		Matrix<float, 3, 4> xform = training_xforms[j];
		Vector3f ploc = pos - xform.col(3);
		float x = ploc.dot(xform.col(0));
		float y = ploc.dot(xform.col(1));
		float z = ploc.dot(xform.col(2));
		if (z > 0.f)
		{
			auto focal = focal_lengths[j];
			// TODO - add a box / plane intersection to stop thomas from murdering me
			if (fabsf(x) - voxel_radius < z / focal.x() * half_resx && fabsf(y) - voxel_radius < z / focal.y() * half_resy)
			{
				count++;
				if (count > 0)
					break;
			}
		}
	}
	if((grid_out[i] < 0) != (count <= 0))
	{
	grid_out[i] = (count > 0) ? 0.f : -1.f;
	}
}