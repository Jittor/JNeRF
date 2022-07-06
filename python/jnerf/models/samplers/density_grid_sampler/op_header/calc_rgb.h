#include "ray_sampler_header.h"
#define gpuErrchk(val) \
    cudaErrorCheck(val, __FILE__, __LINE__, true)

void cudaErrorCheck(cudaError_t err, char* file, int line, bool abort)
{
    if(err != cudaSuccess)
    {
        printf("%s %s %d\n", cudaGetErrorString(err), file, line);
        if(abort) exit(-1);
    }
}

//Users can customize their ways of calculating RGB through rays here.
__device__ void nerf_calculate_rgb_3(const RGBArray rgb, const float dt, const float density, RGBArray& rgb_ray, float& T) {
	const float alpha = 1.f - __expf(-density * dt);
	const float weight = alpha * T;
	rgb_ray += weight * rgb;
	T *= (1.f - alpha);
}

__device__ void nerf_calculate_rgb_grad_3(const RGBArray rgb, const float dt, const float density, RGBArray& rgb_ray2, float& T, RGBArray* rgb_ray, RGBArray* loss_grad, RGBArray& suffix, RGBArray& dloss_by_drgb) {
	const float alpha = 1.f - __expf(-density * dt);
	const float weight = alpha * T;
	rgb_ray2 += weight * rgb;
	T *= (1.f - alpha);
	suffix = *rgb_ray - rgb_ray2; 
	dloss_by_drgb = weight * (*loss_grad);
}


typedef void(*calculate_rgb_grad)(const Array3f, const float, const float, Array3f&, float&, Array3f*, Array3f*, Array3f&, Array3f&);
typedef void(*calculate_rgb)(const Array3f, const float, const float, Array3f&, float&);
__device__ calculate_rgb_grad ccg_ptr = nerf_calculate_rgb_grad_3;
__device__ calculate_rgb cc_ptr = nerf_calculate_rgb_3;
//Use nerf volume rendering formula to calculate the color of every ray
//Calculate the value of each sampling point on ray and calculate integral summation .
//if compacted step equals to that before compacted ,then add background color to the ray color output
void compute_rgbs_fp32_3(
    uint32_t shmem_size,
    cudaStream_t stream,
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,    				//network output width
	const float *network_output, 				//network output
	ENerfActivation rgb_activation, 			//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	uint32_t *__restrict__ numsteps_in,			//rays offset and base counter before compact
	RGBArray *rgb_output, 						//rays rgb output
	uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
	const RGBArray *bg_color_ptr,				//background color 
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE,						//lower bound of step size
	calculate_rgb_grad cc
	);

// Use chain rule to calculate the dloss/dnetwork_output from the dloss/dray_rgb
void compute_rgbs_grad_fp32_3(
    uint32_t shmem_size,
    cudaStream_t stream,
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,					//network output width
	float *__restrict__ dloss_doutput,			//dloss_dnetworkoutput,shape same as network output
	const float *network_output,					//network output
	uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	ENerfActivation rgb_activation,				//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	RGBArray *__restrict__ loss_grad,			//dloss_dRGBoutput
	RGBArray *__restrict__ rgb_ray,				//RGB from forward calculation
	float *__restrict__ density_grid_mean,		//density_grid mean value,
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE,						//lower bound of step size
	calculate_rgb_grad ccg
	);


//Like compute_rgbs, but when inference don't execute compact coords.So this function doesn't use compact data.
void compute_rgbs_inference_fp32_3(
    uint32_t shmem_size,
    cudaStream_t stream,
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,					//network output width
	RGBArray background_color,					//background color
	const float *network_output,					//network output
	ENerfActivation rgb_activation,				//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	uint32_t *__restrict__ numsteps_in,			//rays offset and base counter
	RGBArray *rgb_output,						//rays rgb output
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE,						//lower bound of step size
	calculate_rgb cc
	);

//compute_rgbs in float16 type
void compute_rgbs_fp16_3(
    uint32_t shmem_size,
    cudaStream_t stream,
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,    				//network output width
	const __half *network_output, 				//network output
	ENerfActivation rgb_activation, 			//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	uint32_t *__restrict__ numsteps_in,			//rays offset and base counter before compact
	RGBArray *rgb_output, 						//rays rgb output
	uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
	const RGBArray *bg_color_ptr,				//background color 
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE,						//lower bound of step size
	calculate_rgb cc
	);

//compute_rgbs_grad in float16 type
void compute_rgbs_grad_fp16_3(
    uint32_t shmem_size,
    cudaStream_t stream,
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,					//network output width
	__half *__restrict__ dloss_doutput,			//dloss_dnetworkoutput,shape same as network output
	const __half *network_output,					//network output
	uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	ENerfActivation rgb_activation,				//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	RGBArray *__restrict__ loss_grad,			//dloss_dRGBoutput
	RGBArray *__restrict__ rgb_ray,				//RGB from forward calculation
	float *__restrict__ density_grid_mean,		//density_grid mean value,
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE,						//lower bound of step size
	calculate_rgb_grad ccg
	);


//compute_rgbs_inference in float16 type
void compute_rgbs_inference_fp16_3(
    uint32_t shmem_size,
    cudaStream_t stream,
	const uint32_t n_rays,						//batch total rays number
	BoundingBox aabb,							//boundingbox range
	int padded_output_width,					//network output width
	RGBArray background_color,					//background color
	const __half *network_output,					//network output
	ENerfActivation rgb_activation,				//activation of rgb in output 
	ENerfActivation density_activation,			//activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		//network input,(xyz,dt,dir)
	uint32_t *__restrict__ numsteps_in,			//rays offset and base counter
	RGBArray *rgb_output,						//rays rgb output
	int NERF_CASCADES,							//num of density grid level
	float MIN_CONE_STEPSIZE,						//lower bound of step size
	calculate_rgb cc
	);