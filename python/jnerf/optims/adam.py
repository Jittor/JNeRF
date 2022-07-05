import jittor as jt
from jittor import nn
from jnerf.utils.registry import OPTIMS 
from jnerf.utils.config import init_cfg
import copy

@OPTIMS.register_module()
class Adam(jt.nn.Adam):
    def __init__(self, params, **kwargs):
        super(Adam, self).__init__(params, **kwargs)
    
    @property
    def defaults(self):
        exclude = set(("defaults", "n_step", "pre_step", "step"))
        return copy.deepcopy({ k:v for k, v in self.__dict__.items()
            if k[0] != '_' and k not in exclude and not callable(v) })