import jittor as jt
import numpy as np

from jnerf.dataset import Rays_keys


def enlarge(x: jt.Var, size: int):
    if x.shape[0] < size:
        y = jt.empty([size], x.dtype)
        x.assign(y)


class BoundingBox():
    def __init__(self, min=[0, 0, 0], max=[0, 0, 0]) -> None:
        self.min = np.array(min)
        self.max = np.array(max)

    def inflate(self, amount: float):
        self.min -= amount
        self.max += amount
        pass


def rearrange_render_image(rays, chunk_size=4096):
    # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
    single_image_rays = [getattr(rays, key) for key in Rays_keys]
    val_mask = single_image_rays[-3]

    # flatten each Rays attribute and put on device
    single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # generate N Rays instances
    single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask
