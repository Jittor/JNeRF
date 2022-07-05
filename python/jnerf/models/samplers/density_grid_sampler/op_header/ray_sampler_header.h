#include <atomic>
#include <limits>
#include <stdexcept>
#include "utils/log.h"
#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "cuda_fp16.h"
#include "pcg32.h"
using namespace Eigen;
#if defined(__CUDA_ARCH__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define NGP_PRAGMA_UNROLL _Pragma("unroll")
#define NGP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#define NGP_PRAGMA_UNROLL #pragma unroll
#define NGP_PRAGMA_NO_UNROLL #pragma unroll 1
#endif
#else
#define NGP_PRAGMA_UNROLL
#define NGP_PRAGMA_NO_UNROLL
#endif

#ifdef __NVCC__
#define NGP_HOST_DEVICE __host__ __device__
#else
#define NGP_HOST_DEVICE
#endif

#define TCNN_HOST_DEVICE __host__ __device__
#define TCNN_MIN_GPU_ARCH 70
#define rgb_length 3
typedef Array<float, rgb_length, 1> RGBArray;
static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;
constexpr uint32_t n_threads_linear = 128;

inline NGP_HOST_DEVICE float sign(float x)
{
	return copysignf(1.0, x);
}

enum class EColorSpace : int
{
	Linear,
	SRGB,
	VisPosNeg,
};
NGP_HOST_DEVICE inline float clamp(float val, float lower, float upper)
{
	return val < lower ? lower : (upper < val ? upper : val);
}

template <typename T>
TCNN_HOST_DEVICE void host_device_swap(T &a, T &b)
{
	T c(a);
	a = b;
	b = c;
}

inline __device__ int mip_from_pos(const Vector3f &pos)
{
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(NERF_CASCADES() - 1, max(0, exponent + 1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f &pos)
{
	int mip = mip_from_pos(pos);
	dt *= 2 * NERF_GRIDSIZE();
	if (dt < 1.f)
		return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(NERF_CASCADES() - 1, max(exponent, mip));
}

// triangle
struct Triangle
{
	NGP_HOST_DEVICE Eigen::Vector3f sample_uniform_position(const Vector2f &sample) const
	{
		float sqrt_x = std::sqrt(sample.x());
		float factor0 = 1.0f - sqrt_x;
		float factor1 = sqrt_x * (1.0f - sample.y());
		float factor2 = sqrt_x * sample.y();

		return factor0 * a + factor1 * b + factor2 * c;
	}

	NGP_HOST_DEVICE float surface_area() const
	{
		return 0.5f * Vector3f((b - a).cross(c - a)).norm();
	}

	NGP_HOST_DEVICE Vector3f normal() const
	{
		return (b - a).cross(c - a).normalized();
	}

	// based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
	NGP_HOST_DEVICE float ray_intersect(const Vector3f &ro, const Vector3f &rd, Vector3f &n) const
	{
		Vector3f v1v0 = b - a;
		Vector3f v2v0 = c - a;
		Vector3f rov0 = ro - a;
		n = v1v0.cross(v2v0);
		Vector3f q = rov0.cross(rd);
		float d = 1.0f / rd.dot(n);
		float u = d * -q.dot(v2v0);
		float v = d * q.dot(v1v0);
		float t = d * -n.dot(rov0);
		if (u < 0.0f || u > 1.0f || v < 0.0f || (u + v) > 1.0f || t < 0.0f)
		{
			t = std::numeric_limits<float>::max(); // No intersection
		}
		return t;
	}

	NGP_HOST_DEVICE float ray_intersect(const Vector3f &ro, const Vector3f &rd) const
	{
		Vector3f n;
		return ray_intersect(ro, rd, n);
	}

	// based on https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
	NGP_HOST_DEVICE float distance_sq(const Vector3f &pos) const
	{
		Vector3f v21 = b - a;
		Vector3f p1 = pos - a;
		Vector3f v32 = c - b;
		Vector3f p2 = pos - b;
		Vector3f v13 = a - c;
		Vector3f p3 = pos - c;
		Vector3f nor = v21.cross(v13);

		return
			// inside/outside test
			(sign(v21.cross(nor).dot(p1)) + sign(v32.cross(nor).dot(p2)) + sign(v13.cross(nor).dot(p3)) < 2.0f)
				?
				// 3 edges
				std::min({
					(v21 * clamp(v21.dot(p1) / v21.squaredNorm(), 0.0f, 1.0f) - p1).squaredNorm(),
					(v32 * clamp(v32.dot(p2) / v32.squaredNorm(), 0.0f, 1.0f) - p2).squaredNorm(),
					(v13 * clamp(v13.dot(p3) / v13.squaredNorm(), 0.0f, 1.0f) - p3).squaredNorm(),
				})
				:
				// 1 face
				nor.dot(p1) * nor.dot(p1) / nor.squaredNorm();
	}

	NGP_HOST_DEVICE float distance(const Vector3f &pos) const
	{
		return std::sqrt(distance_sq(pos));
	}

	NGP_HOST_DEVICE bool point_in_triangle(const Vector3f &p) const
	{
		// Move the triangle so that the point becomes the
		// triangles origin
		Vector3f local_a = a - p;
		Vector3f local_b = b - p;
		Vector3f local_c = c - p;

		// The point should be moved too, so they are both
		// relative, but because we don't use p in the
		// equation anymore, we don't need it!
		// p -= p;

		// Compute the normal vectors for triangles:
		// u = normal of PBC
		// v = normal of PCA
		// w = normal of PAB

		Vector3f u = local_b.cross(local_c);
		Vector3f v = local_c.cross(local_a);
		Vector3f w = local_a.cross(local_b);

		// Test to see if the normals are facing the same direction.
		// If yes, the point is inside, otherwise it isn't.
		return u.dot(v) >= 0.0f && u.dot(w) >= 0.0f;
	}

	NGP_HOST_DEVICE Vector3f closest_point_to_line(const Vector3f &a, const Vector3f &b, const Vector3f &c) const
	{
		float t = (c - a).dot(b - a) / (b - a).dot(b - a);
		t = std::max(std::min(t, 1.0f), 0.0f);
		return a + t * (b - a);
	}

	NGP_HOST_DEVICE Vector3f closest_point(Vector3f point) const
	{
		point -= normal().dot(point - a) * normal();

		if (point_in_triangle(point))
		{
			return point;
		}

		Vector3f c1 = closest_point_to_line(a, b, point);
		Vector3f c2 = closest_point_to_line(b, c, point);
		Vector3f c3 = closest_point_to_line(c, a, point);

		float mag1 = (point - c1).squaredNorm();
		float mag2 = (point - c2).squaredNorm();
		float mag3 = (point - c3).squaredNorm();

		float min = std::min({mag1, mag2, mag3});

		if (min == mag1)
		{
			return c1;
		}
		else if (min == mag2)
		{
			return c2;
		}
		else
		{
			return c3;
		}
	}

	NGP_HOST_DEVICE Vector3f centroid() const
	{
		return (a + b + c) / 3.0f;
	}

	NGP_HOST_DEVICE float centroid(int axis) const
	{
		return (a[axis] + b[axis] + c[axis]) / 3;
	}

	NGP_HOST_DEVICE void get_vertices(Vector3f v[3]) const
	{
		v[0] = a;
		v[1] = b;
		v[2] = c;
	}

	Vector3f a, b, c;
};

// bounding box
template <int N_POINTS>
NGP_HOST_DEVICE inline void project(Vector3f points[N_POINTS], const Vector3f &axis, float &min, float &max)
{
	min = std::numeric_limits<float>::infinity();
	max = -std::numeric_limits<float>::infinity();

	NGP_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N_POINTS; ++i)
	{
		float val = axis.dot(points[i]);

		if (val < min)
		{
			min = val;
		}

		if (val > max)
		{
			max = val;
		}
	}
}

struct BoundingBox
{
	NGP_HOST_DEVICE BoundingBox() {}

	NGP_HOST_DEVICE BoundingBox(const Vector3f &a, const Vector3f &b) : min{a}, max{b} {}

	NGP_HOST_DEVICE explicit BoundingBox(const Triangle &tri)
	{
		min = max = tri.a;
		enlarge(tri.b);
		enlarge(tri.c);
	}

	BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end)
	{
		min = max = begin->a;
		for (auto it = begin; it != end; ++it)
		{
			enlarge(*it);
		}
	}

	NGP_HOST_DEVICE void enlarge(const BoundingBox &other)
	{
		min = min.cwiseMin(other.min);
		max = max.cwiseMax(other.max);
	}

	NGP_HOST_DEVICE void enlarge(const Triangle &tri)
	{
		enlarge(tri.a);
		enlarge(tri.b);
		enlarge(tri.c);
	}

	NGP_HOST_DEVICE void enlarge(const Vector3f &point)
	{
		min = min.cwiseMin(point);
		max = max.cwiseMax(point);
	}

	NGP_HOST_DEVICE void inflate(float amount)
	{
		min -= Vector3f::Constant(amount);
		max += Vector3f::Constant(amount);
	}

	NGP_HOST_DEVICE Vector3f diag() const
	{
		return max - min;
	}

	NGP_HOST_DEVICE Vector3f relative_pos(const Vector3f &pos) const
	{
		return (pos - min).cwiseQuotient(diag());
	}

	NGP_HOST_DEVICE Vector3f center() const
	{
		return 0.5f * (max + min);
	}

	NGP_HOST_DEVICE BoundingBox intersection(const BoundingBox &other) const
	{
		BoundingBox result = *this;
		result.min = result.min.cwiseMax(other.min);
		result.max = result.max.cwiseMin(other.max);
		return result;
	}

	NGP_HOST_DEVICE bool intersects(const BoundingBox &other) const
	{
		return !intersection(other).is_empty();
	}

	// Based on the separating axis theorem
	// (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf)
	// Code adapted from a C# implementation at stack overflow
	// https://stackoverflow.com/a/17503268
	NGP_HOST_DEVICE bool intersects(const Triangle &triangle) const
	{
		float triangle_min, triangle_max;
		float box_min, box_max;

		// Test the box normals (x-, y- and z-axes)
		Vector3f box_normals[3] = {
			Vector3f{1.0f, 0.0f, 0.0f},
			Vector3f{0.0f, 1.0f, 0.0f},
			Vector3f{0.0f, 0.0f, 1.0f},
		};

		Vector3f triangle_normal = triangle.normal();
		Vector3f triangle_verts[3];
		triangle.get_vertices(triangle_verts);

		for (int i = 0; i < 3; i++)
		{
			project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
			if (triangle_max < min[i] || triangle_min > max[i])
			{
				return false; // No intersection possible.
			}
		}

		Vector3f verts[8];
		get_vertices(verts);

		// Test the triangle normal
		float triangle_offset = triangle_normal.dot(triangle.a);
		project<8>(verts, triangle_normal, box_min, box_max);
		if (box_max < triangle_offset || box_min > triangle_offset)
		{
			return false; // No intersection possible.
		}

		// Test the nine edge cross-products
		Vector3f edges[3] = {
			triangle.a - triangle.b,
			triangle.a - triangle.c,
			triangle.b - triangle.c,
		};

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				// The box normals are the same as it's edge tangents
				Vector3f axis = edges[i].cross(box_normals[j]);
				project<8>(verts, axis, box_min, box_max);
				project<3>(triangle_verts, axis, triangle_min, triangle_max);
				if (box_max < triangle_min || box_min > triangle_max)
					return false; // No intersection possible
			}
		}

		// No separating axis found.
		return true;
	}

	NGP_HOST_DEVICE Vector2f ray_intersect(const Vector3f &pos, const Vector3f &dir) const
	{
		float tmin = (min.x() - pos.x()) / dir.x();
		float tmax = (max.x() - pos.x()) / dir.x();

		if (tmin > tmax)
		{
			host_device_swap(tmin, tmax);
		}

		float tymin = (min.y() - pos.y()) / dir.y();
		float tymax = (max.y() - pos.y()) / dir.y();

		if (tymin > tymax)
		{
			host_device_swap(tymin, tymax);
		}

		if (tmin > tymax || tymin > tmax)
		{
			return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
		}

		if (tymin > tmin)
		{
			tmin = tymin;
		}

		if (tymax < tmax)
		{
			tmax = tymax;
		}

		float tzmin = (min.z() - pos.z()) / dir.z();
		float tzmax = (max.z() - pos.z()) / dir.z();

		if (tzmin > tzmax)
		{
			host_device_swap(tzmin, tzmax);
		}

		if (tmin > tzmax || tzmin > tmax)
		{
			return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
		}

		if (tzmin > tmin)
		{
			tmin = tzmin;
		}

		if (tzmax < tmax)
		{
			tmax = tzmax;
		}

		return {tmin, tmax};
	}

	NGP_HOST_DEVICE bool is_empty() const
	{
		return (max.array() < min.array()).any();
	}

	NGP_HOST_DEVICE bool contains(const Vector3f &p) const
	{
		return p.x() >= min.x() && p.x() <= max.x() &&
			   p.y() >= min.y() && p.y() <= max.y() &&
			   p.z() >= min.z() && p.z() <= max.z();
	}

	/// Calculate the squared point-AABB distance
	NGP_HOST_DEVICE float distance(const Vector3f &p) const
	{
		return sqrt(distance_sq(p));
	}

	NGP_HOST_DEVICE float distance_sq(const Vector3f &p) const
	{
		return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm();
	}

	NGP_HOST_DEVICE float signed_distance(const Vector3f &p) const
	{
		Vector3f q = (p - min).cwiseAbs() - diag();
		return q.cwiseMax(0.0f).norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f);
	}

	NGP_HOST_DEVICE void get_vertices(Vector3f v[8]) const
	{
		v[0] = {min.x(), min.y(), min.z()};
		v[1] = {min.x(), min.y(), max.z()};
		v[2] = {min.x(), max.y(), min.z()};
		v[3] = {min.x(), max.y(), max.z()};
		v[4] = {max.x(), min.y(), min.z()};
		v[5] = {max.x(), min.y(), max.z()};
		v[6] = {max.x(), max.y(), min.z()};
		v[7] = {max.x(), max.y(), max.z()};
	}

	Vector3f min = Vector3f::Constant(std::numeric_limits<float>::infinity());
	Vector3f max = Vector3f::Constant(-std::numeric_limits<float>::infinity());
};

// other needed structure

struct CameraDistortion
{
	float params[4] = {};
#ifdef __NVCC__
	inline __host__ __device__ bool is_zero() const
	{
		return params[0] == 0.0f && params[1] == 0.0f && params[2] == 0.0f && params[3] == 0.0f;
	}
#endif
};

struct TrainingImageMetadata
{
	// Camera intrinsics and additional data associated with a NeRF training image
	CameraDistortion camera_distortion = {};
	Eigen::Vector2f principal_point = Eigen::Vector2f::Constant(0.5f);
	Eigen::Vector2f focal_length = Eigen::Vector2f::Constant(1000.f);

	// TODO: replace this with more generic float[] of task-specific metadata.
	Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.f);
};

struct NerfPosition
{
	NGP_HOST_DEVICE NerfPosition(const Eigen::Vector3f &pos, float dt) : p{pos} {}
	Eigen::Vector3f p;
};

struct NerfDirection
{
	NGP_HOST_DEVICE NerfDirection(const Eigen::Vector3f &dir, float dt) : d{dir} {}
	Eigen::Vector3f d;
};

struct NerfCoordinate
{
	NGP_HOST_DEVICE NerfCoordinate(const Eigen::Vector3f &pos, const Eigen::Vector3f &dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}
	NGP_HOST_DEVICE void set_with_optional_light_dir(const Eigen::Vector3f &pos, const Eigen::Vector3f &dir, float dt, const Eigen::Vector3f &light_dir, uint32_t stride_in_bytes)
	{
		this->dt = dt;
		this->pos = NerfPosition(pos, dt);
		this->dir = NerfDirection(dir, dt);

		if (stride_in_bytes >= sizeof(Eigen::Vector3f) + sizeof(NerfCoordinate))
		{
			*(Eigen::Vector3f *)(this + 1) = light_dir;
		}
	}
	NGP_HOST_DEVICE void copy_with_optional_light_dir(const NerfCoordinate &inp, uint32_t stride_in_bytes)
	{
		*this = inp;
		if (stride_in_bytes >= sizeof(Eigen::Vector3f) + sizeof(NerfCoordinate))
		{
			*(Eigen::Vector3f *)(this + 1) = *(Eigen::Vector3f *)(&inp + 1);
		}
	}

	NerfPosition pos;
	float dt;
	NerfDirection dir;
};

// struct NerfPayload
// {
// 	Eigen::Vector3f origin;
// 	Eigen::Vector3f dir;
// 	float t;
// 	uint32_t idx;
// 	uint16_t n_steps;
// 	bool alive;
// };

template <typename T>
struct PitchedPtr
{
	TCNN_HOST_DEVICE PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
	TCNN_HOST_DEVICE PitchedPtr(T *ptr, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0) : ptr{ptr + offset}, stride_in_bytes{(uint32_t)(stride_in_elements * sizeof(T) + extra_stride_bytes)} {}

	template <typename U>
	TCNN_HOST_DEVICE explicit PitchedPtr(PitchedPtr<U> other) : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

	TCNN_HOST_DEVICE T *operator()(uint32_t y) const
	{
		return (T *)((const char *)ptr + y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE void operator+=(uint32_t y)
	{
		ptr = (T *)((const char *)ptr + y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE void operator-=(uint32_t y)
	{
		ptr = (T *)((const char *)ptr - y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE explicit operator bool() const
	{
		return ptr;
	}

	T *ptr;
	uint32_t stride_in_bytes;
};


using default_rng_t = pcg32;

template <typename T, uint32_t N_ELEMS>
struct vector_t
{
	TCNN_HOST_DEVICE T &operator[](uint32_t idx)
	{
		return data[idx];
	}

	TCNN_HOST_DEVICE T operator[](uint32_t idx) const
	{
		return data[idx];
	}

	T data[N_ELEMS];
	static constexpr uint32_t N = N_ELEMS;
};
// ==========================
// other needed functions


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

__host__ __device__ inline uint32_t morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

template <typename T>
TCNN_HOST_DEVICE T div_round_up(T val, T divisor)
{
	return (val + divisor - 1) / divisor;
}

constexpr uint32_t batch_size_granularity = 128;

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

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ __host__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
// inline constexpr __device__ __host__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

inline __host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f &focal_length, float cone_angle_constant)
{
	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();

	return cone_angle_constant;
}

inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip)
{
	return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}



// inline __host__ __device__ float calc_dt(float t, float cone_angle)
// {
// 	// TODO: use origin dt
// 	return MIN_CONE_STEPSIZE() * 0.5;
// 	// return clamp(t * cone_angle, MIN_CONE_STEPSIZE(), MAX_CONE_STEPSIZE());
// }

inline __device__ float distance_to_next_voxel(const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{ // dda like step
	Vector3f p = res * pos;
	float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
	float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
	float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
	do
	{
		t += calc_dt(t, cone_angle);
	} while (t < t_target);
	return t;
}

inline __device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip)
{
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= Vector3f::Constant(0.5f);
	pos *= mip_scale;
	pos += Vector3f::Constant(0.5f);

	Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

	uint32_t idx = morton3D(
		clamp(i.x(), 0, (int)NERF_GRIDSIZE() - 1),
		clamp(i.y(), 0, (int)NERF_GRIDSIZE() - 1),
		clamp(i.z(), 0, (int)NERF_GRIDSIZE() - 1));

	return idx;
}

inline __device__ bool density_grid_occupied_at(const Vector3f &pos, const uint8_t *density_grid_bitfield, uint32_t mip)
{
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return density_grid_bitfield[idx / 8 + grid_mip_offset(mip) / 8] & (1 << (idx % 8));
}

inline __device__ float cascaded_grid_at(Vector3f pos, const float *cascaded_grid, uint32_t mip)
{
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx + grid_mip_offset(mip)];
}

inline __device__ float &cascaded_grid_at(Vector3f pos, float *cascaded_grid, uint32_t mip)
{
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx + grid_mip_offset(mip)];
}

inline __device__ Vector3f warp_position(const Vector3f &pos, const BoundingBox &aabb)
{
	// return {logistic(pos.x() - 0.5f), logistic(pos.y() - 0.5f), logistic(pos.z() - 0.5f)};
	// return pos;

	return aabb.relative_pos(pos);
}

inline __device__ Vector3f unwarp_position(const Vector3f &pos, const BoundingBox &aabb)
{
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.min + pos.cwiseProduct(aabb.diag());
}

inline __device__ Vector3f unwarp_position_derivative(const Vector3f &pos, const BoundingBox &aabb)
{
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.diag();
}

inline __device__ Vector3f warp_position_derivative(const Vector3f &pos, const BoundingBox &aabb)
{
	return unwarp_position_derivative(pos, aabb).cwiseInverse();
}

inline __device__ Vector3f warp_direction(const Vector3f &dir)
{
	return (dir + Vector3f::Ones()) * 0.5f;
}

inline __device__ Vector3f unwarp_direction(const Vector3f &dir)
{
	return dir * 2.0f - Vector3f::Ones();
}

inline __device__ Vector3f warp_direction_derivative(const Vector3f &dir)
{
	return Vector3f::Constant(0.5f);
}

inline __device__ Vector3f unwarp_direction_derivative(const Vector3f &dir)
{
	return Vector3f::Constant(2.0f);
}

inline __device__ float warp_dt(float dt)
{
	float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

inline __device__ float unwarp_dt(float dt)
{
	float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}





__device__ inline float random_val(uint32_t seed, uint32_t idx)
{
	pcg32 rng(((uint64_t)seed << 32) | (uint64_t)idx);
	return rng.next_float();
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







template <typename RNG>
inline __host__ __device__ Eigen::Vector2f random_val_2d(RNG &rng)
{
	return {rng.next_float(), rng.next_float()};
}



enum class ENerfActivation : int
{
	None,
	ReLU,
	Logistic,
	Exponential,
};

__host__ __device__ inline float logistic(const float x)
{
	return 1.0f / (1.0f + expf(-x));
}

inline __device__ float network_to_rgb(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::None:
		return val;
	case ENerfActivation::ReLU:
		return val > 0.0f ? val : 0.0f;
	case ENerfActivation::Logistic:
		return logistic(val);
	case ENerfActivation::Exponential:
		return __expf(clamp(val, -10.0f, 10.0f));
	default:
		assert(false);
	}
	return 0.0f;
}
template <typename T>
inline __device__ RGBArray network_to_rgb(const vector_t<T, rgb_length+1> &local_network_output, ENerfActivation activation)
{
	RGBArray rgb;
	for(int i=0;i<rgb_length;i++) {
		rgb[i] = network_to_rgb(float(local_network_output[i]), activation);
	}
	return rgb;
}

inline __device__ float network_to_density(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::None:
		return val;
	case ENerfActivation::ReLU:
		return val > 0.0f ? val : 0.0f;
	case ENerfActivation::Logistic:
		return logistic(val);
	case ENerfActivation::Exponential:
		return __expf(val);
	default:
		assert(false);
	}
	return 0.0f;
}


inline __host__ __device__ float linear_to_srgb(float linear)
{
	if (linear < 0.0031308f)
	{
		return 12.92f * linear;
	}
	else
	{
		return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
	}
}

inline __host__ __device__ Eigen::Array3f linear_to_srgb(const Eigen::Array3f &x)
{
	return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline __host__ __device__ float srgb_to_linear(float srgb)
{
	if (srgb <= 0.04045f)
	{
		return srgb / 12.92f;
	}
	else
	{
		return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
	}
}

inline __host__ __device__ Eigen::Array3f srgb_to_linear(const Eigen::Array3f &x)
{
	return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}
struct LossAndGradient
{
	Eigen::Array3f loss;
	Eigen::Array3f gradient;

	__host__ __device__ LossAndGradient operator*(float scalar)
	{
		return {loss * scalar, gradient * scalar};
	}

	__host__ __device__ LossAndGradient operator/(float scalar)
	{
		return {loss / scalar, gradient / scalar};
	}
};
// inline __device__ LossAndGradient huber_loss(const Array3f &target, const Array3f &prediction, float alpha = 1)
// {
// 	Array3f difference = prediction - target;
// 	Array3f abs_diff = difference.abs();
// 	Array3f square = 0.5f / alpha * difference * difference;
// 	return {
// 		{
// 			abs_diff.x() > alpha ? (abs_diff.x() - 0.5f * alpha) : square.x(),
// 			abs_diff.y() > alpha ? (abs_diff.y() - 0.5f * alpha) : square.y(),
// 			abs_diff.z() > alpha ? (abs_diff.z() - 0.5f * alpha) : square.z(),
// 		},
// 		{
// 			abs_diff.x() > alpha ? (difference.x() > 0 ? 1.0f : -1.0f) : (difference.x() / alpha),
// 			abs_diff.y() > alpha ? (difference.y() > 0 ? 1.0f : -1.0f) : (difference.y() / alpha),
// 			abs_diff.z() > alpha ? (difference.z() > 0 ? 1.0f : -1.0f) : (difference.z() / alpha),
// 		},
// 	};
// }

// __device__ LossAndGradient loss_and_gradient(const Vector3f &target, const Vector3f &prediction)
// {

// 	return huber_loss(target, prediction, 0.1f) / 5.0f;
// }

inline __device__ float network_to_rgb_derivative(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::None:
		return 1.0f;
	case ENerfActivation::ReLU:
		return val > 0.0f ? 1.0f : 0.0f;
	case ENerfActivation::Logistic:
	{
		float density = logistic(val);
		return density * (1 - density);
	};
	case ENerfActivation::Exponential:
		return __expf(clamp(val, -10.0f, 10.0f));
	default:
		assert(false);
	}
	return 0.0f;
}

inline __device__ float network_to_density_derivative(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::None:
		return 1.0f;
	case ENerfActivation::ReLU:
		return val > 0.0f ? 1.0f : 0.0f;
	case ENerfActivation::Logistic:
	{
		float density = logistic(val);
		return density * (1 - density);
	};
	case ENerfActivation::Exponential:
		return __expf(clamp(val, -15.0f, 15.0f));
	default:
		assert(false);
	}
	return 0.0f;
}
