
#pragma once
#include "jt_helper.h"
#include "op.h"
#include "random_util.cuh"
#include "var.h"
enum BasisType {
    // For svox 1 compatibility
    // BASIS_TYPE_RGBA = 0
    BASIS_TYPE_SH = 1,
    // BASIS_TYPE_SG = 2
    // BASIS_TYPE_ASG = 3
    BASIS_TYPE_3D_TEXTURE = 4,
    BASIS_TYPE_MLP = 255,
};
typedef jittor::Var Var;
struct PackedSparseGridSpec {
    float* __restrict__ density_data;
    float* __restrict__ sh_data;
    int32_t* __restrict__ links;

    uint8_t basis_type;
    float* __restrict__ basis_data;

    int32_t* __restrict__ background_links;
    float* __restrict__ background_data;

    int size[3], stride_x;
    int background_reso, background_nlayers;

    int basis_dim, sh_data_dim, basis_reso;
    // float _offset[3];
    // float _scaling[3];
    float* __restrict__ _offset;
    float* __restrict__ _scaling;
    PackedSparseGridSpec(Var* density_data, Var* sh_data, Var* links, Var* _offset, Var* _scaling, Var* background_links, Var* background_data, int basis_dim, uint8_t basis_type, Var* basis_data)
        : density_data(density_data->ptr<float>()),

          sh_data(sh_data->ptr<float>()),
          links(links->ptr<int32_t>()),
          basis_type(basis_type),
          basis_data(basis_data->num != 0 ? basis_data->ptr<float>() : nullptr),
          background_links(background_links->num != 0 ? background_links->ptr<int32_t>() : nullptr),
          background_data(background_data->num != 0 ? background_data->ptr<float>() : nullptr),
          size{(int)links->shape[0],
               (int)links->shape[1],
               (int)links->shape[2]},
          stride_x{(int)links->shape[1] * (int)links->shape[2]},
          background_reso{
              background_links->num != 0 ? (int)background_links->shape[1] : 0,
          },
          background_nlayers{
              background_data->num != 0 ? (int)background_data->shape[1] : 0},
          basis_dim(basis_dim),
          sh_data_dim((int)sh_data->shape[1]),
          basis_reso(basis_data->num != 0 ? basis_data->shape[0] : 0),
          // _offset{_offset->ptr<float>()[0],
          //         _offset->ptr<float>()[1],
          //         _offset->ptr<float>()[2]},
          // _scaling{_scaling->ptr<float>()[0],
          //          _scaling->ptr<float>()[1],
          //          _scaling->ptr<float>()[2]}
          _offset(_offset->ptr<float>()),
          _scaling(_scaling->ptr<float>()) {
        // printf("sh_data_dim%d\n", sh_data_dim);
    }
};
struct PackedRaysSpec {
    // TODO: const
    PackedVar32<float, 2> origins;
    PackedVar32<float, 2> dirs;
    PackedRaysSpec(Var* origins, Var* dirs)
        : origins(PackedVar32<float, 2>(origins)),
          dirs(PackedVar32<float, 2>(dirs)) {}
};

struct SingleRaySpec {
    SingleRaySpec() = default;
    __device__ SingleRaySpec(const float* __restrict__ origin, const float* __restrict__ dir)
        : origin{origin[0], origin[1], origin[2]},
          dir{dir[0], dir[1], dir[2]} {}
    __device__ void set(const float* __restrict__ origin, const float* __restrict__ dir) {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            this->origin[i] = origin[i];
            this->dir[i] = dir[i];
        }
    }

    float origin[3];
    float dir[3];
    float tmin, tmax, world_step;

    float pos[3];
    int32_t l[3];
    RandomEngine32 rng;
};

struct RenderOptions {
    float background_brightness;
    // float step_epsilon;
    float step_size;
    float sigma_thresh;
    float stop_thresh;

    float near_clip;
    bool use_spheric_clip;

    bool last_sample_opaque;

    // bool randomize;
    // float random_sigma_std;
    // float random_sigma_std_background;
    // 32-bit RNG state masks
    // uint32_t _m1, _m2, _m3;

    // int msi_start_layer = 0;
    // int msi_end_layer = 66;
};

struct PackedGridOutputGrads {
    PackedGridOutputGrads(Var* grad_density_out, Var* grad_sh_out, Var* grad_basis_out, Var* grad_background_out, Var* mask_out = nullptr, Var* mask_background_out = nullptr)
        : grad_density_out(grad_density_out->num != 0 ? grad_density_out->ptr<float>() : nullptr),
          grad_sh_out(grad_sh_out->num != 0 ? grad_sh_out->ptr<float>() : nullptr),
          grad_basis_out(grad_basis_out->num != 0 ? grad_basis_out->ptr<float>() : nullptr),
          grad_background_out(grad_background_out->num != 0 ? grad_background_out->ptr<float>() : nullptr),
          mask_out((mask_out != nullptr && mask_out->num != 0 && mask_out->shape[0] > 0) ? mask_out->ptr<bool>() : nullptr),
          mask_background_out((mask_background_out != nullptr && mask_background_out->num != 0 && mask_background_out->shape[0] > 0) ? mask_background_out->ptr<bool>() : nullptr) {
        // printf("111  %p\n", this->grad_density_out);
    }
    float* __restrict__ grad_density_out;
    float* __restrict__ grad_sh_out;
    float* __restrict__ grad_basis_out;
    float* __restrict__ grad_background_out;

    bool* __restrict__ mask_out;
    bool* __restrict__ mask_background_out;
};

#define CAM_FX 0
#define CAM_FY 1
#define CAM_CX 2
#define CAM_CY 3
#define CAM_WIDTH 4
#define CAM_HEIGHT 5
#define CAM_NDCX 6
#define CAM_NDCY 7
struct PackedCameraSpec {
    PackedCameraSpec(Var* c2w, Var* cam_info)
        : c2w(PackedVar32<float, 2>(c2w)),
          cam_info(cam_info->ptr<float>()) {}
    const PackedVar32<float, 2> c2w;
    const float* __restrict__ cam_info;

    __device__ float info(int i) const {
        return cam_info[i];
    }
    __device__ int width() const {
        return int(cam_info[CAM_WIDTH]);
    }
    __device__ int height() const {
        return int(cam_info[CAM_HEIGHT]);
    }
    // float fx;
    // float fy;
    // float cx;
    // float cy;
    // int width;
    // int height;

    // float ndc_coeffx;
    // float ndc_coeffy;
};