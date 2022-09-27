import jittor as jt
from .c_class import CameraSpec
from .global_header import proj_path

class grid_weight_render(jt.Function):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, data, cam:CameraSpec,step_size,stop_thresh,last_sample_opaque,offset,scaling,grid_weight_out):
        cam_info = jt.array([cam.fx,cam.fy,cam.cx,cam.cy,cam.width,cam.height,cam.ndc_coeffx,cam.ndc_coeffy])
        cam_size = jt.empty([0,cam.width,cam.height])
        grid_weight_out, = jt.code(inputs=[data,offset,scaling,cam_info,cam_size,cam.c2w], outputs=[grid_weight_out], cuda_header='#include "misc_kernel.h"', cuda_src=f"""
        @alias(grid_data,in0)
        @alias(offset,in1)
        @alias(scaling,in2)
        @alias(cam_info,in3)
        @alias(cam_size,in4)
        @alias(cam_c2w,in5)
        @alias(grid_weight_out,out0)
        const int Q = cam_size_shape1*cam_size_shape2;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);
        grid_weight_render_kernel<<<blocks, MISC_CUDA_THREADS>>>(
        PackedVar32<float, 3>(grid_data),
        {{cam_c2w,cam_info}},
        {step_size},
        {stop_thresh},
        {'true' if last_sample_opaque else 'false'},
        offset_p,
        scaling_p,
        PackedVar32<float,3>(grid_weight_out),
        grid_data_p);
        CUDA_CHECK_ERRORS;
        """)
        grid_weight_out.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        # grid_weight_out.sync(True)
        return grid_weight_out

    def grad(self, x):
        assert False
        return None
