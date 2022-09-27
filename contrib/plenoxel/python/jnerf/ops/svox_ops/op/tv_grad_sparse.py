import jittor as jt

from .global_header import proj_path
class tv_grad_sparse(jt.Function):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, links, data, rand_cells, mask_out,start_dim,end_dim,scale,ignore_edge,ignore_last_z, grad_data,):


        grad_data, mask_out = jt.code(inputs=[links, data, rand_cells], outputs=[grad_data, mask_out], cuda_header='#include "loss_kernel.h"', cuda_src=f"""
        @alias(links,in0)
        @alias(data,in1)
        @alias(rand_cells,in2)

        @alias(grad_data,out0)
        @alias(mask_out,out1)

        int nl = rand_cells_shape0;
        size_t Q = rand_cells_shape0 * size_t({end_dim} - {start_dim});
        

        const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        tv_grad_sparse_kernel<<<blocks, cuda_n_threads>>>(
                PackedVar32<int32_t,3>(links),
                PackedVar64<float,2>(data),
                rand_cells_p,
                { start_dim},
                { end_dim},
                {scale} / nl,
                Q,
                {'true'if ignore_edge else 'false'},
                {'true' if ignore_last_z else 'false'},
                // Output
                (mask_out_shape0 > 0) ? mask_out_p : nullptr,
                grad_data_p);
        CUDA_CHECK_ERRORS;

        """)
        grad_data.compile_options = {
            f"FLAGS: -I{proj_path}": 1}

        return grad_data,mask_out

    def grad(self, x):
        assert False
        return None
