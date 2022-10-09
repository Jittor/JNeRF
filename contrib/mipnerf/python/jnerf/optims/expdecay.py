import jittor as jt
from jittor import nn
from jnerf.utils.registry import OPTIMS 
from jnerf.utils.config import init_cfg

@OPTIMS.register_module()
class ExpDecay(jt.nn.Optimizer):
    def __init__(self,nested_optimizer:jt.nn.Optimizer,decay_start:int,decay_interval:int,decay_base:float,decay_end:int=None):
        self.base_lr=nested_optimizer.lr
        self._nested_optimizer=nested_optimizer
        self.decay_start=decay_start
        self.decay_interval=decay_interval
        self.decay_base=decay_base
        if decay_end is None:
            self.decay_end=10000000
        else:
            self.decay_end=decay_end
        self.steps=0
        self.m_learning_rate_factor=1
    def step(self, loss=None):
        if self.steps>=self.decay_start and (self.steps-self.decay_start)%self.decay_interval==0 and self.steps<=self.decay_end:
            self.m_learning_rate_factor*=self.decay_base
        self._nested_optimizer.lr=self.base_lr*self.m_learning_rate_factor
        self._nested_optimizer.step(loss)
        self.steps += 1
    def zero_grad(self):
        return self._nested_optimizer.zero_grad()
    
    def backward(self, loss, retain_graph=False):
        return self._nested_optimizer.backward(loss, retain_graph)