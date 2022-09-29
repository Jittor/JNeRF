import jittor as jt
import numpy as np

def enlarge(x: jt.Var, size: int):
    if x.shape[0] < size:
        y = jt.empty([size],x.dtype)
        x.assign(y)

class BoundingBox():
    def __init__(self,min=[0,0,0],max=[0,0,0]) -> None:
        self.min=np.array(min)
        self.max=np.array(max)
    def inflate(self,amount:float):
        self.min-=amount
        self.max+=amount
        pass


def volume_render(sigmas, z_vals, raw_rgbs, dirs):
    '''
    Volume rendering function

    Inputs:
            sigmas: NeRF network output raw sigmas (N_rays, N_samples)
            z_vals: depths of the sampled positions (N_rays, N_samples)
            raw_rgbs: NeRF network output raw rgb (N_rays, N_samples,3)
            dirs: ray directions(non-unit)(N_rays,3)
    Outputs:
            rgbs: final rays' rgb (N_rays,3)
            depths: final depth map (N_rays)
            weights: weights of each sample (N_rays,N_samples)

    '''
    # breakpoint()

    deltas = z_vals[:, 1:]-z_vals[:, :-1]
    deltas_inf = 1e10 * jt.ones_like(deltas[:, :1])
    deltas = jt.concat([deltas, deltas_inf], -1)
    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * jt.norm(dirs.unsqueeze(1), dim=-1)
    raw_rgbs = jt.sigmoid(raw_rgbs)
    # compute alpha by the formula
    alphas = 1-jt.exp(-deltas*(jt.nn.relu(sigmas)))  # (N_rays, N_samples_)
    alphas_shifted = jt.concat([jt.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1)
    weights = alphas*jt.cumprod(alphas_shifted, -1)[:, :-1]  # formula(3) Ti
    rgbs = jt.sum(weights.unsqueeze(-1)*raw_rgbs, -2)
    depths = jt.sum(weights*z_vals, -1)
    return rgbs, depths, weights