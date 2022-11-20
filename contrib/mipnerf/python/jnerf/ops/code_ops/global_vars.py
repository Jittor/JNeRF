import os
import sys
import jittor as jt
jt.flags.use_cuda = 1

global_headers = """
#include "pcg32.h"
namespace jittor {
extern int global_var1;
extern pcg32 rng;
}
"""

global_src = """
namespace jittor {
int global_var1 = 123;
pcg32 rng{1337};
}
"""

proj_path = os.path.join(os.path.dirname(__file__), '..', 'op_include')
# if sys.platform == "linux":
proj_options = { f'FLAGS: -I"{proj_path}/eigen" -I"{proj_path}/include" -I"{proj_path}/pcg32" -I"{proj_path}/../op_header" -DGLOBAL_VAR --extended-lambda --expt-relaxed-constexpr': 1 }
# elif sys.platform == "windows":
#     proj_options = { f"FLAGS: /I {proj_path}/eigen /I {proj_path}/include /I {proj_path}/pcg32 /I {proj_path}/../op_header -DGLOBAL_VAR --extended-lambda --expt-relaxed-constexpr": 1 }
gv = None
def get_global_var():
    global gv, global_headers, global_src
    gv = jt.code([1], int, 
    cuda_header=global_headers+global_src, 
    cuda_src="""
    printf("Init the global vars here here here.!!");
    exit(0);
    """)
    gv.compile_options = proj_options
    gv.sync()

if sys.platform == "linux":
    gv = jt.code([1], int, 
        cuda_header=global_headers+global_src, 
        cuda_src="""
        printf("Init the global vars here here here.!!");
        exit(0);
    """)
    gv.compile_options = proj_options
    gv.sync()
