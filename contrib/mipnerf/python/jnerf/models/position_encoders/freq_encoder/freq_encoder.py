import os
import jittor as jt
from jittor import Function
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import ENCODERS

@ENCODERS.register_module()
class FrequencyEncoder(Function):
    def __init__(self, multires, include_input = True, input_dims = 3, log_sampling = True, periodic_fns = [jt.sin, jt.cos]):
        self.cfg = get_cfg()
        self.using_fp16 = self.cfg.fp16
        self.multires = multires
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs
        
        if self.log_sampling:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def execute(self, x):
        res=jt.concat([fn(x) for fn in self.embed_fns], -1)
        if self.using_fp16:
            res=res.float16()
        return res