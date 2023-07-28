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

jt.profiler.start()
gv = jt.code([1], int, 
    cuda_header=global_src, 
    cuda_src="""
""")
gv.compile_options = proj_options
gv.sync()
jt.profiler.stop()

if os.name == "nt":
    dll_name = jt.profiler.report()[-1][-10].replace(".cc", "")
    proj_options[f'FLAGS: -l{dll_name} '] = 1