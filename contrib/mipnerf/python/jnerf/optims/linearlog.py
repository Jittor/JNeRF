import jittor as jt
from jittor import nn
from jnerf.utils.registry import OPTIMS
from jnerf.utils.config import init_cfg
import numpy as np


@OPTIMS.register_module()
class LinearLog(jt.nn.Optimizer):
    def __init__(self, nested_optimizer: jt.nn.Optimizer, start_lr=5e-4, end_lr=5e-6, max_steps=40000, lr_delay_steps=0,
                 lr_delay_mult=1):
        self.start_lr = nested_optimizer.lr
        self.nested_optimizer = nested_optimizer
        self.end_lr = end_lr
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.steps = 0
        self.m_learning_rate_factor = 1

    def step(self, loss=None):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * jt.sin(
                0.5 * jt.array(np.pi) * jt.safe_clip(self.steps / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = jt.safe_clip(self.steps / self.max_steps, 0, 1)
        log_lerp = jt.exp(jt.log(self.start_lr) * (1 - t) + jt.log(self.end_lr) * t)
        self.nested_optimizer.lr = delay_rate * log_lerp
        self.nested_optimizer.step(loss)
        self.steps += 1

    def zero_grad(self):
        return self.nested_optimizer.zero_grad()

    def backward(self, loss, retain_graph=False):
        return self.nested_optimizer.backward(loss, retain_graph)
