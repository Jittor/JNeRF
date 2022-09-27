import jittor as jt


class PlenOptim(jt.nn.Optimizer):
    def __init__(self, p_density, p_sh, lr_sigma, lr_sh):
        params = [self.add_param(p_density, lr_sigma),
                  self.add_param(p_sh, lr_sh)]
        super().__init__(params, lr=0)
        self.lr_sigma = lr_sigma
        self.lr_sh = lr_sh
        self.density_id = id(p_density)
        self.sh_id = id(p_sh)

    def update_lr(self, lr_sigma, lr_sh):
        self.lr_sh = lr_sh
        self.lr_sigma = lr_sigma
        self.param_groups[0]['lr'] = self.lr_sigma
        self.param_groups[1]['lr'] = self.lr_sh

    def add_param(self, param, lr):
        return {'params': [param], 'lr': lr}

    def find_grad_by_id(self, id: int) -> jt.Var:
        if id not in self._grad_map:
            self._build_grad_map()
            if id not in self._grad_map:
                raise RuntimeError(
                    "This variable is not managed by this optimizer")
        return self._grad_map[id]


class PlenOptimSGD(jt.nn.SGD):
    def __init__(self, p_density, p_sh, lr_sigma, lr_sh, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        params = [self.add_param(p_density, lr_sigma),
                  self.add_param(p_sh, lr_sh)]
        super().__init__(params, lr=0)
        self.lr_sigma = lr_sigma
        self.lr_sh = lr_sh
        self.density_id = id(p_density)
        self.sh_id = id(p_sh)

    def update_lr(self, lr_sigma, lr_sh):
        self.lr_sh = lr_sh
        self.lr_sigma = lr_sigma
        self.param_groups[0]['lr'] = self.lr_sigma
        self.param_groups[1]['lr'] = self.lr_sh

    def add_param(self, param, lr):
        return {'params': [param], 'lr': lr}


class PlenOptimRMSprop(jt.nn.RMSprop):
    def __init__(self, p_density, p_sh, lr_sigma, lr_sh, alpha_sigma, alpha_sh, eps=1e-8):
        params = [self.add_param(p_density, lr_sigma, alpha_sigma),
                  self.add_param(p_sh, lr_sh, alpha_sh)]
        super().__init__(params, lr=0)
        self.alpha_sigma=alpha_sigma
        self.alpha_sh=alpha_sh
        self.lr_sigma = lr_sigma
        self.lr_sh = lr_sh
        self.density_id = id(p_density)
        self.sh_id = id(p_sh)

    def add_param(self, param, lr, alpha):
        return {'params': [param], 'lr': lr, 'alpha': alpha}
    
    def update_lr(self, lr_sigma, lr_sh,alpha_sigma,alpha_sh):
        self.lr_sh = lr_sh
        self.lr_sigma = lr_sigma
        self.alpha_sigma=alpha_sigma
        self.alpha_sh=alpha_sh
        self.param_groups[0]['lr'] = self.lr_sigma
        self.param_groups[1]['lr'] = self.lr_sh
        self.param_groups[0]['alpha']=self.alpha_sigma
        self.param_groups[1]['alpha']=self.alpha_sh
    
    def step(self, loss=None, retain_graph=False):
        super().step(loss, retain_graph)
        # self.param_groups[0]['params'][0].assign(jt.clamp(self.param_groups[0]['params'][0],min_v=1e-9))
        # self.param_groups[1]['params'][0].assign(jt.clamp(self.param_groups[1]['params'][0],min_v=1e-9))

