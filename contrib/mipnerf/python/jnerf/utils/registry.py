class Registry:
    def __init__(self):
        self._modules = {}

    def register_module(self, name=None, module=None):
        def _register_module(module):
            key = name
            if key is None:
                key = module.__name__
            assert key not in self._modules, f"{key} is already registered."
            self._modules[key] = module
            return module

        if module is not None:
            return _register_module(module)

        return _register_module

    def get(self, name):
        assert name in self._modules, f"{name} is not registered."
        return self._modules[name]


def build_from_cfg(cfg, registry, **kwargs):
    if isinstance(cfg, str):
        return registry.get(cfg)(**kwargs)
    elif isinstance(cfg, dict):
        args = cfg.copy()
        args.update(kwargs)
        obj_type = args.pop('type')
        obj_cls = registry.get(obj_type)
        try:
            module = obj_cls(**args)
        except TypeError as e:
            if "<class" not in str(e):
                e = f"{obj_cls}.{e}"
            raise TypeError(e)

        return module
    elif isinstance(cfg, list):
        from jittor import nn
        return nn.Sequential([build_from_cfg(c, registry, **kwargs) for c in cfg])
    elif cfg is None:
        return None
    else:
        raise TypeError(f"type {type(cfg)} not support")


DATASETS = Registry()
ENCODERS = Registry()
NETWORKS = Registry()
SAMPLERS = Registry()
LOSSES = Registry()
OPTIMS = Registry()
SCHEDULERS = Registry()
