#include"ray_sampler_header.h"


__global__ void rays_sampler(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const Vector3f *__restrict__ rays_o,
	const Vector3f *__restrict__ rays_d,
	const uint8_t *__restrict__ density_grid,
	const float cone_angle_constant,
	const TrainingImageMetadata *__restrict__ metadata,
	const uint32_t *__restrict__ imgs_index,
	uint32_t *__restrict__ ray_counter,
	uint32_t *__restrict__ numsteps_counter,
	uint32_t *__restrict__ ray_indices_out,
	uint32_t *__restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const Matrix<float, 3, 4> *training_xforms,
	float near_distance,
	default_rng_t rng

)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	// i (0,n_rays)
	if (i >= n_rays)
		return;
	uint32_t img = imgs_index[i];
	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	float max_level = 1.0f; // Multiply by 2 to ensure 50% of training is at max level
	// float max_level = max_level_rand_training ? (0.6 * 2.0f) : 1.0f;
	const Matrix<float, 3, 4> xform = training_xforms[img];
	const Vector2f focal_length = metadata[img].focal_length;
	const Vector2f principal_point = metadata[img].principal_point;
	const Vector3f light_dir_warped = warp_direction(metadata[img].light_dir);
	const CameraDistortion camera_distortion = metadata[img].camera_distortion;
	Vector3f ray_o = rays_o[i];
	Vector3f ray_d = rays_d[i];

	Vector2f tminmax = aabb.ray_intersect(ray_o, ray_d);
	float cone_angle = calc_cone_angle(ray_d.dot(xform.col(2)), focal_length, cone_angle_constant);
	// // The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x() = fmaxf(tminmax.x(), near_distance);

	float startt = tminmax.x();
	// // TODO:change
	startt += calc_dt(startt, cone_angle) * random_val(rng);

	Vector3f idir = ray_d.cwiseInverse();

	// // first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;
	Vector3f pos;
	while (aabb.contains(pos = ray_o + t * ray_d) && j < NERF_STEPS())
	{
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip))
		{
			++j;
			t += dt;
		}
		else
		{
			uint32_t res = NERF_GRIDSIZE() >> mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d, idir, res);
		}
	}

	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps); // first entry in the array is a counter
	if (base + numsteps > max_samples)
	{
		// printf("over max sample!!!!!!!!!!!!!!\n");
		numsteps_out[2 * i + 0] = 0;
		numsteps_out[2 * i + 1] = base;
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);
	ray_indices_out[i] = ray_idx;
	// TODO:
	numsteps_out[2 * i + 0] = numsteps;
	numsteps_out[2 * i + 1] = base;
	if (j == 0)
	{
		ray_indices_out[i] = -1;
		return;
	}
	Vector3f warped_dir = warp_direction(ray_d);
	t = startt;
	j = 0;
	while (aabb.contains(pos = ray_o + t * ray_d) && j < numsteps)
	{
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip))
		{

			coords_out(j)->set_with_optional_light_dir(warp_position(pos, aabb), warped_dir, warp_dt(dt), light_dir_warped, coords_out.stride_in_bytes);
			++j;
			t += dt;
		}
		else
		{
			uint32_t res = NERF_GRIDSIZE() >> mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d, idir, res);
		}
	}
}

