"""
Implements image encoders
"""
import os
import jittor as jt
from jittor import nn
from jittor import models
import numpy as np
from jnerf.ops.code_ops.global_vars import global_headers,proj_options
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import ENCODERS

@ENCODERS.register_module()
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_first_pool = True
        self.model = models.resnet34(pretrained=True)

    def execute(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        feats1 = self.model.relu(x)

        if self.use_first_pool:
            feats1 = self.model.maxpool(feats1)
        feats2 = self.model.layer1(feats1)
        feats3 = self.model.layer2(feats2)
        feats4 = self.model.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = nn.interpolate(
                latents[i],
                latent_sz,
                mode='bilinear',
                align_corners=True,
            )
        latents = jt.concat(latents, dim=1)
        return latents
