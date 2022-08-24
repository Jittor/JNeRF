import jittor as jt
from jittor import nn
from jnerf.utils.registry import OPTIMS 
from jnerf.utils.config import init_cfg
import copy

@OPTIMS.register_module()
class EMA(jt.nn.Optimizer):
    def __init__(self, params, decay):
        super().__init__(params, lr=0)
        self.decay = decay
        self.steps = 0

        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            for p in pg["params"]:
                values.append(p.copy())

    @property
    def defaults(self):
        exclude = set(("defaults", "n_step", "pre_step", "step"))
        return  copy.deepcopy( { k:v for k, v in self.__dict__.items()
            if k[0] != '_' and k not in exclude and not callable(v) })
            
    def ema_step(self, loss=None):
        assert(loss is None)
        self.steps += 1
        ema_debias_old = 1-self.decay**(self.steps-1)
        ema_debias_new = 1/(1-self.decay**self.steps)
        for pg in self.param_groups:

            for p, v in zip(pg["params"], pg["values"]):
                p.update((((1-self.decay)*p +
                         self.decay*v*ema_debias_old)*ema_debias_new).detach())
                v.update(p.detach())
        self.zero_grad()