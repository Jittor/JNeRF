

import jittor as jt
import numpy as np
# Basis types (copied from C++ data_spec.hpp)
BASIS_TYPE_SH = 1
BASIS_TYPE_3D_TEXTURE = 4
BASIS_TYPE_MLP = 255


class GridOutputGrads():


    def __init__(self):
        self.grad_density_out = None
        self.grad_sh_out = None
        self.grad_basis_out = None
        self.grad_background_out = None
        self.mask_out = None
        self.mask_background_out = None



class SparseGridSpec():
    '''

    '''

    def __init__(self):
        self.density_data = None
        self.sh_data = None
        self._links = None
        self._offset = None
        self._scaling = None
        self._background_links = None
        self.background_data = None
        self.basis_dim: int = None
        self.basis_type: int = None  # uint8
        self.basis_data = None

    def test(self):
        bs = 8962370
        self.density_data = jt.rand(bs, 1)
        self.sh_data = jt.rand(bs, 27)
        self._links = jt.randint(0, 10, [256, 256, 256])
        self._offset = jt.array([0.5000, 0.5000, 0.5000])
        self._scaling = jt.array([0.5000, 0.5000, 0.5000])
        self.background_links = jt.empty([0, 1])
        self.basis_data = jt.empty([0, 1])
        self.background_data = jt.empty([0, 1])
        self.basis_dim = 9
        self.basis_type = 1  # SH
      

class RaysSpec():
    '''

    '''

    def __init__(self):
        self.origins = None
        self.dirs = None

    def test(self):
        bs = 5000
        self.origins = jt.rand(5000, 3)
        self.dirs = jt.rand(5000, 3)
     

class RenderOptions():
    def __init__(self):
        self.background_brightness: float = None
        self.step_size: float = None
        self.sigma_thresh: float = None
        self.stop_thresh: float = None
        self.near_clip: float = None
        self.use_spheric_clip: bool = None
        self.last_sample_opaque: bool = None

    def test(self):
        self.background_brightness = 1.0
        self.step_size = 0.5
        self.sigma_thresh = 1e-8
        self.stop_thresh = 1e-7
        self.last_sample_opaque = 0
        self.near_clip = 0.0
        self.use_spheric_clip = 0


class GridOutputGrads():
    def __init__(self):
        self.grad_density_out = None
        self.grad_sh_out = None
        self.grad_basis_out = None
        self.grad_background_out = None
        self.mask_out = None
        self.mask_background_out = None

class CameraSpec():
    def __init__(self) -> None:
        self.c2w = None
        self.fx  = 0.
        self.fy = 0.
        self.cx = 0.
        self.cy = 0.
        self.width = 0
        self.height = 0
        self.ndc_coeffx = 0.
        self.ndc_coeffy = 0.
