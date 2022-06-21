import jittor as jt
from jittor import nn
import os
import pathlib
import numpy as np
import jittor_utils
from jnerf.ops.code_ops.global_vars import global_headers, proj_options, ngp_suffix, fn_mapping

cuda_header = '''
#include "fully_fused_mlp_header.h"
'''

class FullyFusedMlp_weight(jt.Function):
    def __init__(self, weights, check_mid="0", output_activation="Activation::None"):
        self.input = None
        self.outputs = []
        self.shapes = []
        self.dtypes = []
        self.weights_grad = []
        self.max_dim = 0
        self.weights = weights
        self.width = 0
        self.output_intermediate = None
        con_weights = []
        self.code_path = pathlib.Path(__file__+"/../op_header").resolve()
        user_jittor_path = os.path.join(jittor_utils.cache_path, "ngp_cache")
        self.so_name = os.path.join(user_jittor_path, fn_mapping["fm"]+ngp_suffix)
        for i in range(len(weights)):
            if i == 0:
                self.weight_shape0 = weights[0].shape[0]
                self.width = weights[0].shape[1]
            if i == len(weights) - 1:
                assert weights[i].shape[0] == weights[i-1].shape[1]
                self.output_shape1 = weights[i].shape[1]
                if weights[i].shape[1] < 16: 
                    weights[i] = jt.concat([weights[i], jt.zeros((weights[i].shape[0], 16 - weights[i].shape[1]))], -1).float16()
            if i !=0 and i != len(weights) - 1:
                assert weights[i].shape[0] == weights[i].shape[1]
                assert weights[i-1].shape[1] == weights[i].shape[0]
            con_weights.append(weights[i].transpose(1,0).reshape(-1))
        self.output_width = self.width*(len(weights)-1)
        self.con_weights = jt.concat(con_weights, -1)
        self.first_stride = weights[0].shape[0] * self.width
        self.output_activation = output_activation
        self.check_mid = check_mid
        
    def execute(self, a, con_weights):
        if a.shape[0] == 0:
            return jt.empty([0]).float16()
        self.shapes = []
        self.dtypes = []
        assert a.shape[1] == self.weights[0].shape[0]
        self.output_intermediate = None
        cuda_src = f'''
        @alias(input, in0)
        @alias(weights, in1)
        @alias(output_intermediate, out1)
        @alias(output, out0)
        cudaStream_t stream = 0;
        mlp_fused_forward_func(
            {self.width}, 
            Activation::ReLU, 
            false,
            stream,
            {self.output_activation},
            weights_p,
            input_p,
            output_intermediate_p,
            output_p,
            {len(self.weights) - 2},
            input_shape0,
            input_shape1,
            {self.weight_shape0},
            {self.width},
            input_shape0,
            16
        );
        '''
        self.input = a
        self.padded_shape0 = (self.input.shape[0] + 127) // 128 * 128 - a.shape[0]
        if self.padded_shape0 > 0:
            self.padded_input = jt.concat([a, jt.zeros((self.padded_shape0, a.shape[1])).float16()], 0)
        else:
            self.padded_input = self.input
        self.outputs, self.output_intermediate = jt.code([(self.padded_input.shape[0], 16), (self.padded_input.shape[0] * (len(self.weights) - 1), self.width)], [a.dtype, a.dtype], [self.padded_input, con_weights], cuda_header=cuda_header, cuda_src=cuda_src)
        self.outputs.compile_options = {f"FLAGS: -I{self.code_path} -Xlinker {self.so_name} ":1}
        self.con_weights = con_weights
        return self.outputs[:self.input.shape[0]]

    def grad(self, grads):
        if self.padded_shape0 != 0:
            grads = jt.concat([grads, jt.zeros((self.padded_shape0, grads.shape[1])).float16()], 0)
        self.grads = grads
        need_last = 1 if self.width == self.input.shape[1] else 0
        cuda_src = f'''
        @alias(grad, in0)
        @alias(weights, in1)
        @alias(output_intermediate, in2) // need or not.
        @alias(temp, out1)
        @alias(output, out0)
        cudaStream_t stream = 0;
        // LOGir << {self.width * self.input.shape[1]};
        mlp_fused_backward_func(
            {self.width},
            Activation::ReLU,
            stream,
            weights_p, // weights_first_layer
            &weights_p[{self.width * self.input.shape[1]}], // remained weights
            grad_p,
            temp_p,
            output_intermediate_p,
            output_p,
            {len(self.weights) - 2},
            grad_shape1,
            grad_shape0,
            {need_last}
        );
        '''
        output, grad_temps = jt.code([(self.padded_input.shape[0], self.input.shape[1]), ((len(self.weights)-1) * self.padded_input.shape[0],  self.width)], [self.input.dtype, self.input.dtype], [grads.transpose(), self.con_weights, self.output_intermediate], cuda_header=cuda_header, cuda_src=cuda_src)
        output.compile_options = {f"FLAGS: -I{self.code_path} -Xlinker {self.so_name} ":1}
        if self.check_mid == "1":
            self.grad_temps = grad_temps
        if not need_last:
            output = jt.matmul(grad_temps[self.padded_input.shape[0] * (len(self.weights)-2):], self.con_weights[:self.first_stride].reshape(self.width, self.weight_shape0))
        wt = []
        for i in range(len(self.weights)):
            if i == 0:
                if os.environ.get("FUSE_TRANSPOSE", "0") == "1":
                    new_weight = jt.cublas.ops.cublas_acc_matmul(grad_temps[self.padded_input.shape[0] * (len(self.weights) - 2 - i): ], self.padded_input, 1, 0)
                else:
                    new_weight = jt.cublas.ops.cublas_acc_matmul(self.padded_input, grad_temps[self.padded_input.shape[0] * (len(self.weights) - 2 - i): ], 1, 0).transpose()
                wt.append(new_weight.reshape(-1))
            elif i == len(self.weights) - 1:
                if os.environ.get("FUSE_TRANSPOSE", "0") == "1":
                    new_weight = jt.cublas.ops.cublas_acc_matmul(grads, self.output_intermediate[self.padded_input.shape[0] * (len(self.weights) - 2):], 1, 0)
                else:
                    new_weight = jt.cublas.ops.cublas_acc_matmul(self.output_intermediate[self.padded_input.shape[0] * (len(self.weights) - 2):], grads, 1, 0).transpose()
                new_weight[self.output_shape1:,:] = 0
                wt.append(new_weight.reshape(-1))
            else:
                if os.environ.get("FUSE_TRANSPOSE", "0") == "1":
                    new_weight = jt.cublas.ops.cublas_acc_matmul(grad_temps[self.padded_input.shape[0] * (len(self.weights) - 2 - i): self.padded_input.shape[0] * (len(self.weights) - 1 - i)], self.output_intermediate[self.padded_input.shape[0] * (len(self.weights) - 2 - i):self.padded_input.shape[0] * (len(self.weights) - 1 - i)], 1, 0)
                else:
                    new_weight = jt.cublas.ops.cublas_acc_matmul(self.output_intermediate[self.padded_input.shape[0] * (len(self.weights) - 2 - i):self.padded_input.shape[0] * (len(self.weights) - 1 - i)], grad_temps[self.padded_input.shape[0] * (len(self.weights) - 2 - i): self.padded_input.shape[0] * (len(self.weights) - 1 - i)], 1, 0).transpose()
                wt.append(new_weight.reshape(-1))
        
        return output[:self.input.shape[0]], jt.concat(wt, -1)

if __name__ == "__main__":
    pass
