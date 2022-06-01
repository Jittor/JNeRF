#include"density_grid_sampler_header.h"

__global__ void ema_grid_samples_nerf(const uint32_t n_elements,
									  float decay,
									  float *__restrict__ grid_out,
									  const float *__restrict__ grid_in)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;

	float importance = grid_in[i];

	// float ema_debias_old = 1 - (float)powf(decay, count);
	// float ema_debias_new = 1 - (float)powf(decay, count+1);

	// float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
	// grid_out[i] = filtered_val;

	// Maximum instead of EMA allows capture of very thin features.
	// Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

	float prev_val = grid_out[i];
	float val = (prev_val < 0.f) ? prev_val : fmaxf(prev_val * decay, importance);
	grid_out[i] = val;
}