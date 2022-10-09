#include"ray_sampler_header.h"


__device__ float unwarp_dt(float dt, int NERF_CASCADES, float MIN_CONE_STEPSIZE)
{
	float max_stepsize = MIN_CONE_STEPSIZE * (1 << (NERF_CASCADES - 1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE) + MIN_CONE_STEPSIZE;
}

template <typename TYPE>
__global__ void compute_rgbs(
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,    				//network output width
	const TYPE *network_output, 				//network output
	ENerfActivation rgb_activation, 			//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	uint32_t *__restrict__ numsteps_in,			//rays offset and base counter before compact
	Array3f *rgb_output, 						//rays rgb output
	uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
	const Array3f *bg_color_ptr,				//background color 
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE						//lower bound of step size
	)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays)
	{
		return;
	}
	Array3f background_color=bg_color_ptr[i];
	uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
	uint32_t base = numsteps_compacted_in[i * 2 + 1];
	if (numsteps == 0)
	{
		rgb_output[i] = background_color;
		return;
	}
	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float EPSILON = 1e-4f;

	Array3f rgb_ray = Array3f::Zero();

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt, NERF_CASCADES, MIN_CONE_STEPSIZE);

		float density = network_to_density(float(local_network_output[3]), density_activation);

		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray += weight * rgb;

		T *= (1.f - alpha);
		network_output += padded_output_width;
		coords_in += 1;
	}

	if (compacted_numsteps == numsteps_in[i * 2 + 0])
	{
		rgb_ray += T * background_color;
	}

	rgb_output[i] = rgb_ray;
}

template <typename TYPE>
__global__ void compute_rgbs_grad(
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,					//network output width
	TYPE *__restrict__ dloss_doutput,			//dloss_dnetworkoutput,shape same as network output
	const TYPE *network_output,					//network output
	uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	ENerfActivation rgb_activation,				//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	Array3f *__restrict__ loss_grad,			//dloss_dRGBoutput
	Array3f *__restrict__ rgb_ray,				//RGB from forward calculation
	float *__restrict__ density_grid_mean,		//density_grid mean value,
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE						//lower bound of step size
	)
{

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays)
	{
		return;
	}
	float loss_scale = 128;
	loss_scale /= n_rays;
	uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
	uint32_t base = numsteps_compacted_in[i * 2 + 1];

	coords_in += base;
	network_output += base * padded_output_width;
	dloss_doutput += base * padded_output_width;
	loss_grad += i;
	rgb_ray += i;

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = *density_grid_mean < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	float T = 1.f;
	uint32_t compacted_numsteps = 0;
	Array3f rgb_ray2 = Array3f::Zero();
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		
		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		float dt = unwarp_dt(coords_in.ptr->dt, NERF_CASCADES, MIN_CONE_STEPSIZE);
		float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray2 += weight * rgb;
		T *= (1.f - alpha);

		const Array3f suffix = *rgb_ray - rgb_ray2; 
		const Array3f dloss_by_drgb = weight * (*loss_grad);

		vector_t<TYPE, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x() * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y() * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z() * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		float dloss_by_dmlp = density_derivative * (dt * (*loss_grad).matrix().dot((T * rgb - suffix).matrix()));
		local_dL_doutput[3] = loss_scale * dloss_by_dmlp + (float(local_network_output[3]) < 0 ? -output_l1_reg_density : 0.0f);
		*(vector_t<TYPE, 4> *)dloss_doutput = local_dL_doutput;

		network_output += padded_output_width;
		dloss_doutput += padded_output_width;
		coords_in += 1;
	}
}


template <typename TYPE>
__global__ void compute_rgbs_inference(
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,					//network output width
	Array3f background_color,					//background color
	const TYPE *network_output,					//network output
	ENerfActivation rgb_activation,				//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	uint32_t *__restrict__ numsteps_in,			//rays offset and base counter
	Array3f *__restrict__ rgb_output,						//rays rgb output
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE,					//lower bound of step size
	float* __restrict__ alpha_output
	)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_rays)
	{
		return;
	}

	uint32_t numsteps = numsteps_in[i * 2 + 0];
	uint32_t base = numsteps_in[i * 2 + 1];
	if (numsteps == 0)
	{
		rgb_output[i] = Array3f::Zero();
		alpha_output[i] = 0;
		return;
	}
	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float EPSILON = 1e-4f;

	Array3f rgb_ray = Array3f::Zero();

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt, NERF_CASCADES, MIN_CONE_STEPSIZE);

		float density = network_to_density(float(local_network_output[3]), density_activation);

		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray += weight * rgb;

		T *= (1.f - alpha);
		network_output += padded_output_width;
		coords_in += 1;
	}
	rgb_output[i] = rgb_ray;
	alpha_output[i] = 1-T;
}
