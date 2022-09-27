
import jittor as jt

jt.flags.use_cuda=1
from .global_header import proj_path
class dilate(jt.Function):
    def __init__(self,) -> None:
        super().__init__()

    def execute(self, grid) -> None:
        result = jt.empty(grid.shape).astype(grid.dtype)

        result, = jt.code(inputs=[grid],outputs=[result],cuda_header='#include "misc_kernel.h"',cuda_src=f"""
        @alias(grid,in0)

        @alias(result,out0)
        int Q= grid_shape0*grid_shape1*grid_shape2;
         const int blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);

         dilate_kernel<<<blocks, MISC_CUDA_THREADS>>>(
            PackedVar32<bool,3>(grid),
                   // Output
            PackedVar32<bool,3>( result)
          );
        """)
        result.compile_options = {
            f"FLAGS: -I{proj_path}": 1}
        return result
