#include <atomic>
#include <limits>
#include <stdexcept>
#include "utils/log.h"
#include <stdio.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "pcg32.h"
#define TCNN_HOST_DEVICE __host__ __device__
#define NGP_HOST_DEVICE __host__ __device__
using namespace Eigen;

using default_rng_t = pcg32;
constexpr uint32_t n_threads_linear = 128;
template <typename T>
TCNN_HOST_DEVICE T div_round_up(T val, T divisor)
{
	return (val + divisor - 1) / divisor;
}
template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements)
{
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}
template <typename K, typename T, typename... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types... args)
{
	if (n_elements <= 0)
	{
		return;
	}
	kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>((uint32_t)n_elements, args...);
}
struct NerfPosition
{
	NGP_HOST_DEVICE NerfPosition(const Eigen::Vector3f &pos, float dt) : p{pos} {}

	Eigen::Vector3f p;
};

__device__ float warp_dt(float dt)
{
	float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}
template <typename RNG>
inline __host__ __device__ float random_val(RNG &rng)
{
	return rng.next_float();
}
template <typename RNG>
inline __host__ __device__ Eigen::Vector3f random_val_3d(RNG &rng)
{
	return {rng.next_float(), rng.next_float(), rng.next_float()};
}
__host__ __device__ inline uint32_t morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

struct BoundingBox
{
	NGP_HOST_DEVICE BoundingBox() {}

	NGP_HOST_DEVICE BoundingBox(const Eigen::Vector3f &a, const Eigen::Vector3f &b) : min{a}, max{b} {}

	// From triangle
	// NGP_HOST_DEVICE explicit BoundingBox(const Triangle& tri) {
	// 	min = max = tri.a;
	// 	enlarge(tri.b);
	// 	enlarge(tri.c);
	// }

	// From iterators
	// BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end) {
	// 	min = max = begin->a;
	// 	for (auto it = begin; it != end; ++it) {
	// 		enlarge(*it);
	// 	}
	// }

	NGP_HOST_DEVICE void enlarge(const BoundingBox &other)
	{
		min = min.cwiseMin(other.min);
		max = max.cwiseMax(other.max);
	}

	// NGP_HOST_DEVICE void enlarge(const Triangle& tri) {
	// 	enlarge(tri.a);
	// 	enlarge(tri.b);
	// 	enlarge(tri.c);
	// }

	NGP_HOST_DEVICE void enlarge(const Eigen::Vector3f &point)
	{
		min = min.cwiseMin(point);
		max = max.cwiseMax(point);
	}

	// NGP_HOST_DEVICE void inflate(float amount) {
	// 	min -= Eigen::Vector3f::Constant(amount);
	// 	max += Eigen::Vector3f::Constant(amount);
	// }

	NGP_HOST_DEVICE Eigen::Vector3f diag() const
	{
		return max - min;
	}

	NGP_HOST_DEVICE Eigen::Vector3f relative_pos(const Eigen::Vector3f &pos) const
	{
		return (pos - min).cwiseQuotient(diag());
	}

	// NGP_HOST_DEVICE Eigen::Vector3f center() const {
	// 	return 0.5f * (max + min);
	// }

	// NGP_HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
	// 	BoundingBox result = *this;
	// 	result.min = result.min.cwiseMax(other.min);
	// 	result.max = result.max.cwiseMin(other.max);
	// 	return result;
	// }

	// NGP_HOST_DEVICE bool intersects(const BoundingBox& other) const {
	// 	return !intersection(other).is_empty();
	// }

	// // NGP_HOST_DEVICE bool intersects(const Triangle& triangle) const {
	// // 	return !intersection(BoundingBox{triangle}).is_empty();
	// // }

	// // Based on the separating axis theorem
	// // (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf)
	// // Code adapted from a C# implementation at stack overflow
	// // https://stackoverflow.com/a/17503268
	// NGP_HOST_DEVICE bool intersects(const Triangle& triangle) const {
	// 	float triangle_min, triangle_max;
	// 	float box_min, box_max;

	// 	// Test the box normals (x-, y- and z-axes)
	// 	Eigen::Vector3f box_normals[3] = {
	// 		Eigen::Vector3f{1.0f, 0.0f, 0.0f},
	// 		Eigen::Vector3f{0.0f, 1.0f, 0.0f},
	// 		Eigen::Vector3f{0.0f, 0.0f, 1.0f},
	// 	};

	// 	Eigen::Vector3f triangle_normal = triangle.normal();
	// 	Eigen::Vector3f triangle_verts[3];
	// 	triangle.get_vertices(triangle_verts);

	// 	for (int i = 0; i < 3; i++) {
	// 		project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
	// 		if (triangle_max < min[i] || triangle_min > max[i])
	// 			return false; // No intersection possible.
	// 	}

	// 	Eigen::Vector3f verts[8];
	// 	get_vertices(verts);

	// 	// Test the triangle normal
	// 	float triangle_offset = triangle_normal.dot(triangle.a);
	// 	project<8>(verts, triangle_normal, box_min, box_max);
	// 	if (box_max < triangle_offset || box_min > triangle_offset) {
	// 		return false; // No intersection possible.
	// 	}

	// 	// Test the nine edge cross-products
	// 	Eigen::Vector3f edges[3] = {
	// 		triangle.a - triangle.b,
	// 		triangle.a - triangle.c,
	// 		triangle.b - triangle.c,
	// 	};

	// 	for (int i = 0; i < 3; i++) {
	// 		for (int j = 0; j < 3; j++) {
	// 			// The box normals are the same as it's edge tangents
	// 			Eigen::Vector3f axis = edges[i].cross(box_normals[j]);
	// 			project<8>(verts, axis, box_min, box_max);
	// 			project<3>(triangle_verts, axis, triangle_min, triangle_max);
	// 			if (box_max < triangle_min || box_min > triangle_max)
	// 				return false; // No intersection possible
	// 		}
	// 	}

	// 	// No separating axis found.
	// 	return true;
	// }

	// NGP_HOST_DEVICE Eigen::Vector2f ray_intersect(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir) const {
	// 	float tmin = (min.x() - pos.x()) / dir.x();
	// 	float tmax = (max.x() - pos.x()) / dir.x();

	// 	if (tmin > tmax) tcnn::host_device_swap(tmin, tmax);

	// 	float tymin = (min.y() - pos.y()) / dir.y();
	// 	float tymax = (max.y() - pos.y()) / dir.y();

	// 	if (tymin > tymax) tcnn::host_device_swap(tymin, tymax);

	// 	if ((tmin > tymax) || (tymin > tmax))
	// 		return { 100000.0f, 100000.0f };

	// 	if (tymin > tmin)
	// 		tmin = tymin;

	// 	if (tymax < tmax)
	// 		tmax = tymax;

	// 	float tzmin = (min.z() - pos.z()) / dir.z();
	// 	float tzmax = (max.z() - pos.z()) / dir.z();

	// 	if (tzmin > tzmax) tcnn::host_device_swap(tzmin, tzmax);

	// 	if ((tmin > tzmax) || (tzmin > tmax))
	// 		return { 100000.0f, 100000.0f };

	// 	if (tzmin > tmin)
	// 		tmin = tzmin;

	// 	if (tzmax < tmax)
	// 		tmax = tzmax;

	// 	return { tmin, tmax };
	// }

	// NGP_HOST_DEVICE bool is_empty() const {
	// 	return (max.array() < min.array()).any();
	// }

	// NGP_HOST_DEVICE bool contains(const Eigen::Vector3f& p) const {
	// 	return
	// 		p.x() >= min.x() && p.x() <= max.x() &&
	// 		p.y() >= min.y() && p.y() <= max.y() &&
	// 		p.z() >= min.z() && p.z() <= max.z();
	// }

	// /// Calculate the squared point-AABB distance
	// NGP_HOST_DEVICE float distance(const Eigen::Vector3f& p) const {
	// 	return sqrt(distance_sq(p));
	// }

	// NGP_HOST_DEVICE float distance_sq(const Eigen::Vector3f& p) const {
	// 	return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm();
	// }

	// NGP_HOST_DEVICE float signed_distance(const Eigen::Vector3f& p) const {
	// 	Eigen::Vector3f q = (p - min).cwiseAbs() - diag();
	// 	return q.cwiseMax(0.0f).norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f);
	// }

	// NGP_HOST_DEVICE void get_vertices(Eigen::Vector3f v[8]) const {
	// 	v[0] = {min.x(), min.y(), min.z()};
	// 	v[1] = {min.x(), min.y(), max.z()};
	// 	v[2] = {min.x(), max.y(), min.z()};
	// 	v[3] = {min.x(), max.y(), max.z()};
	// 	v[4] = {max.x(), min.y(), min.z()};
	// 	v[5] = {max.x(), min.y(), max.z()};
	// 	v[6] = {max.x(), max.y(), min.z()};
	// 	v[7] = {max.x(), max.y(), max.z()};
	// }

	Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
	Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
};

__device__ Eigen::Vector3f warp_position(const Eigen::Vector3f &pos, const BoundingBox &aabb)
{
	return aabb.relative_pos(pos);
}

__host__ __device__ inline float logistic(const float x)
{
	return 1.0f / (1.0f + expf(-x));
}
enum class ENerfActivation : int
{
	None,
	ReLU,
	Logistic,
	Exponential,
};
__device__ float network_to_density(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::None:
		return val;
	case ENerfActivation::ReLU:
		return val > 0.0f ? val : 0.0f;
		// TODO:
	case ENerfActivation::Logistic:
		return logistic(val);
	case ENerfActivation::Exponential:
		return __expf(val);
	default:
		assert(false);
	}
	return 0.0f;
}


template <typename T>
inline __device__ T warp_reduce(T val)
{
#pragma unroll
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		val += __shfl_xor_sync(0xffffffff, val, offset);
	}

	return val;
}

template <typename T, typename T_OUT, typename F>
__global__ void block_reduce(
	const uint32_t n_elements,
	const F fun,
	const T *__restrict__ input,
	T_OUT *__restrict__ output,
	const uint32_t n_blocks)
{
	const uint32_t sum_idx = blockIdx.x / n_blocks;
	const uint32_t sub_blocks_idx = blockIdx.x % n_blocks;

	const uint32_t i = threadIdx.x + sub_blocks_idx * blockDim.x;
	const uint32_t block_offset = sum_idx * n_elements;

	static __shared__ T_OUT sdata[32];

	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	using T_DECAYED = std::decay_t<T>;

	T_OUT val;
	val = 0; // tmp
	if (std::is_same<T_DECAYED, __half>::value || std::is_same<T_DECAYED, ::half>::value)
	{
		if (i < n_elements)
		{
			::half vals[8];
			*(int4 *)&vals[0] = *((int4 *)input + i + block_offset);
			val = fun((T)vals[0]) + fun((T)vals[1]) + fun((T)vals[2]) + fun((T)vals[3]) + fun((T)vals[4]) + fun((T)vals[5]) + fun((T)vals[6]) + fun((T)vals[7]);
		}
		else
		{
			val = 0;
		}
	}
	else
	if (std::is_same<T_DECAYED, float>::value)
	{
		if (i < n_elements)
		{
			float4 vals = *((float4 *)input + i + block_offset);
			val = fun((T)vals.x) + fun((T)vals.y) + fun((T)vals.z) + fun((T)vals.w);
		}
		else
		{
			val = 0;
		}
	}
	else if (std::is_same<T_DECAYED, double>::value)
	{
		if (i < n_elements)
		{
			double2 vals = *((double2 *)input + i + block_offset);
			val = fun((T)vals.x) + fun((T)vals.y);
		}
		else
		{
			val = 0;
		}
	}
	else
	{
		assert(false);
		return;
	}

	val = warp_reduce(val);

	if (lane == 0)
		sdata[wid] = val;

	__syncthreads();

	if (wid == 0)
	{
		val = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
		val = warp_reduce(val);

		if (lane == 0)
		{
			atomicAdd(&output[sum_idx], val);
		}
	}
}
//TODO: remove or use
uint32_t reduce_sum_workspace_size(uint32_t n_elements)
{
	return n_blocks_linear(n_elements);
}






__host__ __device__ inline uint32_t expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}


inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
	return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}