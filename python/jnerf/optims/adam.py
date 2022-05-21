import jittor as jt
from jittor import nn
from jnerf.utils.registry import OPTIMS 
from jnerf.utils.config import init_cfg

@OPTIMS.register_module()
class Adam(jt.nn.Adam):
    def __init__(self, params, **kwargs):
        super(Adam, self).__init__(params, **kwargs)