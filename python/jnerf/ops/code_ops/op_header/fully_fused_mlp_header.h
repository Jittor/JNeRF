#pragma once
#undef out
#include <map>
#include <type_traits>
#include <stdint.h>
#include <algorithm>
#include <stdexcept>
#include <mma.h>
#define RM MatrixLayout::kRowMajor
#define CM MatrixLayout::kColumnMajor
typedef enum MatrixLayout {
kColumnMajor,
kRowMajor
} MatrixLayout;

typedef enum Activation {
    ReLU,
    Exponential,
    Sine,
    Sigmoid,
    Squareplus,
    Softplus,
    None,
} Activation;

void mlp_fused_backward_func(
    int WIDTH, 
    Activation ACTIVATION,
    cudaStream_t stream,
    __half* weights_first_layer,
    __half* weights,
    __half* dL_doutput,
    __half* temps,
    __half* forward,
    __half* dL_dinput,
    const uint32_t n_hidden_matmuls,
    int grad_shape0,
    int grad_shape1,
    int need_last
);


void mlp_fused_forward_func(
    int WIDTH,
    Activation ACTIVATION,
    bool INFERENCE,
    cudaStream_t stream,
    Activation output_activation,
    __half* weights,
    __half* input,
    __half* output_intermediate,
    __half* output,
    const uint32_t n_hidden_layers,
    int input_shape0,
    int input_shape1,
    int weights_shape0,
    int weights_shape1,
    int output_shape0,
    int output_shape1
);
