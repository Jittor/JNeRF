import os
import sys
import jittor as jt
jt.flags.use_cuda = 1

export_gloabl = ''
import_global = 'extern'
if sys.platform == "win32":
    export_gloabl = '__declspec(dllexport)'
    import_global = '__declspec(dllimport) extern'

global_headers = f"""
#include "pcg32.h"
namespace jittor {{
EXTERN_LIB int global_var1;
EXTERN_LIB pcg32 rng;
}}
"""

global_decl_headers = f"""
#include "pcg32.h"
namespace jittor {{
{export_gloabl} int global_var1;
{export_gloabl} pcg32 rng;
}}
"""

global_src = f"""
#include "pcg32.h"
namespace jittor {{
{export_gloabl} int global_var1 = 123;
{export_gloabl} pcg32 rng{{1337}};
}}
"""

proj_path = os.path.join(os.path.dirname(__file__), '..', 'op_include')
if sys.platform == "linux":
    proj_options = { f"FLAGS: -I{proj_path}/eigen -I{proj_path}/include -I{proj_path}/pcg32 -I{proj_path}/../op_header -DGLOBAL_VAR --extended-lambda --expt-relaxed-constexpr": 1 }
else:
    proj_options = { f'FLAGS: -I"{proj_path}/eigen" -I"{proj_path}/include" -I"{proj_path}/pcg32" -I"{proj_path}/../op_header" -DGLOBAL_VAR --extended-lambda --expt-relaxed-constexpr': 1 }

# def start_up():
with jt.profile_scope() as pr:
    gv = jt.code([1], int, 
        cuda_header=global_src, 
        cuda_src="""
        printf("ggg");
    """)
    gv.compile_options = proj_options
    gv.sync()

# start_up()
proj_options[r'FLAGS: -l"C:\Users\penghy\.cache\jittor\jt1.3.5\cl\py3.8.13\Windows-10-10.xf5\12thGenIntelRCx5f\master\cu11.7.64\jit\code__IN_SIZE_0__OUT_SIZE_1__out0_dim_1__out0_type_int32__HEADER___include__pcg32_h__names___hash_f7de0131cfb188a9_op" '] = 1