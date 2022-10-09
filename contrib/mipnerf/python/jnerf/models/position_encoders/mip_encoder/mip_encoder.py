import os
import jittor as jt
from jittor import Function
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
from jnerf.utils.registry import ENCODERS
from jnerf.utils.miputils import *

@ENCODERS.register_module()
class MIPEncoder(Function):
    def __init__(self,min_deg_point,max_deg_point,using_fp16=False) :
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        # if using_fp16:
        #     self.grad_type='float16'
        # else:
        #     self.grad_type='float32'
        # header_path = os.path.join(os.path.dirname(__file__), 'op_header')
        # proj_options[f"FLAGS: -I{header_path}"]=1
    
    def execute(self, x) :
        samples_enc = integrated_pos_enc(
          x,
          self.min_deg_point,
          self.max_deg_point,
        )
        return samples_enc
