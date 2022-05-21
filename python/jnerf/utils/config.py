# reference maskrcnn_benchmark default
from collections import OrderedDict
from jnerf.utils.general import check_file
import os
import yaml
import copy
from importlib import import_module
import sys
import inspect

__all__ = ["get_cfg","init_cfg","save_cfg","print_cfg"]
BASE_KEY = "_base_"
COVER_KEY = "_cover_"
IGNORE_KEY = "_ignore_"

class Config(OrderedDict):
    def __init__(self, *args):
        super().__init__()
        if len(args)==1:
            self.load_from_file(args[0])
        else:
            assert(len(args)==0)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return None

    def __setattr__(self, name, value):
        self[name] = value

    # Load config file to a Dict without considering BASE_KEY
    @staticmethod
    def _load_dict_from_file_no_base(filename):
        if check_file(filename,ext=[".yaml"]):
            with open(filename, "r") as f:
                cfg = yaml.safe_load(f.read())
        elif check_file(filename,ext=[".py"]):
            f_dir = os.path.dirname(filename)
            f_name = os.path.basename(filename)
            module_name = f_name[:-3]

            # temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, f_dir)
            # Config._validate_py_syntax(filename)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            # delete imported module
            del sys.modules[module_name]
        else:
            assert(False), "unsupported config type."
        return cfg

    # Load config file to a Dict with considering BASE_KEY
    @staticmethod
    def _load_dict_from_file(filename):
        cfg = Config._load_dict_from_file_no_base(filename)
        cfg_dir = os.path.dirname(filename)
        if (BASE_KEY in cfg):
            if (isinstance(cfg[BASE_KEY], list)):
                base_filenames = cfg[BASE_KEY]
            else:
                assert(isinstance(cfg[BASE_KEY], str))
                base_filenames = [cfg[BASE_KEY]]
            cfg_base = {}
            for bfn in base_filenames:
                Config.merge_dict_b2a(cfg_base, Config._load_dict_from_file(os.path.join(cfg_dir, bfn)))
            cfg.pop(BASE_KEY)
            Config.merge_dict_b2a(cfg_base, cfg)
            cfg = cfg_base
        return cfg

    @staticmethod
    def merge_dict_b2a(a, b):
        def clear_cover_key(a):
            if (not isinstance(a, dict)):
                return a
            out = copy.deepcopy(a)
            if (COVER_KEY in out):
                out.pop(COVER_KEY)
            for k, v in out.items():
                out[k] = clear_cover_key(v)
            return out

        assert(isinstance(a, dict))
        assert(isinstance(b, dict))
        if (COVER_KEY in b):
            a.clear()
            temp = clear_cover_key(copy.deepcopy(b))
            a.update(temp)
            return
        for k, v in b.items():
            if ((not(k in a)) or (isinstance(v, dict) and v.get(COVER_KEY, False)) or (not isinstance(v, dict)) or (not isinstance(a[k], dict))):
                a[k] = clear_cover_key(copy.deepcopy(v))
            else:
                Config.merge_dict_b2a(a[k], v)

    def load_from_file(self, filename):
        cfg = Config._load_dict_from_file(filename)
        self.clear()
        self.update(self.dfs(cfg))
        if (self.name is None):
            self.name = os.path.splitext(os.path.basename(filename))[0]
        if self.work_dir is None:
            self.work_dir = f"work_dirs/{self.name}"
    
    def dfs(self, cfg_other):
        if isinstance(cfg_other,dict):
            now_cfg = Config()
            for k,d in cfg_other.items():
                if (inspect.ismodule(d)):
                    continue
                now_cfg[k]=self.dfs(d)
        elif isinstance(cfg_other,list):
            now_cfg = [self.dfs(d) for d in cfg_other if (not inspect.ismodule(d))]
        else:
            now_cfg = copy.deepcopy(cfg_other)
        return now_cfg
    
    def dump(self):
        """convert Config to dict"""
        now = dict()
        for k,d in self.items():
            if isinstance(d,Config):
                d = d.dump()
            if isinstance(d,list):
                d = [dd.dump() if isinstance(dd,Config) else dd for dd in d]
            now[k]=d
        return now

'''
TODO: _cover_ between sibling nodes 
Configs forms a tree like structure through '_base_', while '_cover_' will only work on child nodes, and will not work between sibling nodes.
Maybe we need a cover works on sibling nodes, like:
    _base_: ['a.yaml', '@b.yaml']
means each attribute with a depth of 1 in b.yaml will have the attribute '_cover' before merge into a.yaml, and '@@b.yaml' means a depth of 2, etc.
'''

_cfg = Config()

def init_cfg(filename):
    print("Loading config from: ",filename)
    _cfg.load_from_file(filename)

def get_cfg():
    return _cfg

def update_cfg(**kwargs):
    _cfg.update(kwargs)

def save_cfg(save_file):
    with open(save_file,"w") as f:
        f.write(yaml.dump(_cfg.dump()))

def print_cfg():
    data  =  yaml.dump(_cfg.dump())
    # TODO: data keys are not sorted
    print(data) 