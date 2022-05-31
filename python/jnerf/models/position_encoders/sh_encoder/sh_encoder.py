import os
import jittor as jt
from jittor import Function
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import ENCODERS

@ENCODERS.register_module()
class SHEncoder(Function):
    def __init__(self) :
        self.cfg = get_cfg()
        using_fp16 = self.cfg.fp16
        self.num_elements=4194304
        self.m_n_padded_output_dims=16
        self.m_sh_degree=4
        self.m_n_to_pad=0
        if using_fp16:
            self.grad_type='float16'
        else:
            self.grad_type='float32'
        header_path = os.path.join(os.path.dirname(__file__), 'op_header')
        proj_options[f"FLAGS: -I{header_path}"]=1
        self.out_dim=self.m_n_padded_output_dims
    
    def execute(self,x) :
        self.num_elements=x.shape[0]

        output=jt.code((self.num_elements,16),self.grad_type,[x],cuda_header='#include "SphericalEncode.h"',cuda_src=f"""
  
       #define grad_t out_type

        uint32_t num_elements=in0_shape0;
        uint32_t m_n_padded_output_dims={self.m_n_padded_output_dims};
        uint32_t m_sh_degree={self.m_sh_degree};
        uint32_t m_n_to_pad={self.m_n_to_pad};
       
        cudaStream_t stream=0;
    
        PitchedPtr<const float> inputs={{in0_p,in0_shape1}};
		PitchedPtr<grad_t> outputs={{out_p,out_shape1}};
		float* dy_dx = nullptr;
        linear_kernel(kernel_sh<grad_t>, 0, stream,
			num_elements,
			m_sh_degree,
			m_n_to_pad,
			inputs,
            outputs,
			dy_dx
		);
        """)
        output.compile_options=proj_options
        return output

    def grad(self,grad_x):
        return None
