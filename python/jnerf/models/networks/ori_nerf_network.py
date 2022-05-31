from turtle import pos, position
import jittor as jt
from jittor import nn, init
import os
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
from jnerf.ops.code_ops.fully_fused_mlp import FullyFusedMlp_weight

@NETWORKS.register_module()
class OriginNeRFNetworks(nn.Module):
    def __init__(self, D=8, W=256, skips=[4]):
        super(OriginNeRFNetworks, self).__init__()
        self.D=D
        self.W=W
        self.skips=skips

        self.cfg = get_cfg()
        self.using_fp16 = self.cfg.fp16
        self.pos_encoder = build_from_cfg(self.cfg.encoder.pos_encoder, ENCODERS)
        self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.pos_encoder.out_dim, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.pos_encoder.out_dim, W) for i in range(D-1)])
        self.views_linears = nn.ModuleList([nn.Linear(self.dir_encoder.out_dim + W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        self.set_fp16()

    def execute(self, pos_input, dir_input):  
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(pos_input, dir_input)
        else:
            return self.execute_(pos_input, dir_input)

    def execute_(self, pos_input, dir_input):   
        dir_input = self.dir_encoder(dir_input)
        pos_input = self.pos_encoder(pos_input)

        h = pos_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jt.nn.relu(h)
            if i in self.skips:
                h = jt.concat([pos_input, h], -1)
        alpha = self.alpha_linear(h)

        feature = self.feature_linear(h)
        h = jt.concat([feature, dir_input], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = jt.nn.relu(h)
        rgb = self.rgb_linear(h)
        outputs = jt.concat([rgb, alpha], -1)
        
        return outputs

    def density(self, pos_input):  
        pos_input = self.pos_encoder(pos_input)
        h = pos_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jt.nn.relu(h)
            if i in self.skips:
                h = jt.concat([pos_input, h], -1)
        alpha = self.alpha_linear(h)
        return alpha

    def set_fp16(self):
        if self.using_fp16:
            self.pts_linears.float16()
            self.views_linears.float16()
            self.feature_linear.float16()
            self.alpha_linear.float16()
            self.rgb_linear.float16()
            self.pos_encoder.float16()
            self.dir_encoder.float16()