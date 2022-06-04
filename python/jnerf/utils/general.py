import jittor as jt 
import time 
import warnings
import numpy as np 
import random
import os 
import glob
from functools import partial
from six.moves import map, zip
import numpy

def to_jt_var(data):
    """
        convert data to jt_array
    """
    def _to_jt_var(data):
        if isinstance(data,(list,tuple)):
            data =  [_to_jt_var(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_to_jt_var(d) for k,d in data.items()}
        elif isinstance(data,numpy.ndarray):
            data = jt.array(data)
        elif not isinstance(data,(int,float,str,np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    
    return _to_jt_var(data) 

def sync(data,reduce_mode="mean",to_numpy=True):
    """
        sync data and convert data to numpy
    """
    def _sync(data):
        if isinstance(data,(list,tuple)):
            data =  [_sync(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_sync(d) for k,d in data.items()}
        elif isinstance(data,jt.Var):
            if jt.in_mpi:
                data = data.mpi_all_reduce(reduce_mode)
            if to_numpy:
                data = data.numpy()
        elif not isinstance(data,(int,float,str,np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    
    return _sync(data) 

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.ndim == 1:
        ret = jt.full((count,), fill,dtype=data.dtype)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = jt.full(new_size, fill,dtype=data.dtype)
        ret[inds, :] = data
    return ret
    
def parse_losses(losses):
    _losses = dict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, jt.Var):
            _losses[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            _losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    total_loss = sum(_value for _key, _value in _losses.items() if 'loss' in _key)
    return total_loss, _losses


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.seed(seed)

def current_time():
    return time.asctime( time.localtime(time.time()))

def check_file(file,ext=None):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    if ext:
        if not os.path.splitext(file)[1] in ext:
            # warnings.warn(f"the type of {file} must be in {ext}")
            return False
    return True

def build_file(work_dir,prefix):
    """ build file and makedirs the file parent path """
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix)>0:
        work_dir = os.path.join(work_dir,prefix)
    os.makedirs(work_dir,exist_ok=True)
    file = os.path.join(work_dir,file_name)
    return file 

def check_interval(step,step_interval):
    if step is None or step_interval is None:
        return False 
    if step % step_interval==0:
        return True 
    return False 

def check_dir(work_dir):
    os.makedirs(work_dir,exist_ok=True)


def list_files(file_dir):
    if os.path.isfile(file_dir):
        return [file_dir]

    filenames = []
    for f in os.listdir(file_dir):
        ff = os.path.join(file_dir, f)
        if os.path.isfile(ff):
            filenames.append(ff)
        elif os.path.isdir(ff):
            filenames.extend(list_files(ff))

    return filenames


def is_img(f):
    ext = os.path.splitext(f)[1]
    return ext.lower() in [".jpg",".bmp",".jpeg",".png","tiff"]

def list_images(img_dir):
    img_files = []
    for img_d in img_dir.split(","):
        if len(img_d)==0:
            continue
        if not os.path.exists(img_d):
            raise f"{img_d} not exists"
        img_d = os.path.abspath(img_d)
        img_files.extend([f for f in list_files(img_d) if is_img(f)])
    return img_files

def search_ckpt(work_dir):
    files = glob.glob(os.path.join(work_dir,"checkpoints/ckpt_*.pkl"))
    if len(files)==0:
        return None
    files = sorted(files,key=lambda x:int(x.split("_")[-1].split(".pkl")[0]))
    return files[-1]  

def is_win():
    import platform
    return platform.system() == "Windows"

def get_data_o():
    import shutil
    import pathlib
    import jnerf
    work_dir = pathlib.Path(jnerf.__file__+"/../").resolve()
    user_jittor_path = os.path.expanduser("~/.cache/jittor/ngp_cache")
    data_o_path = os.path.join(work_dir, "utils", "data_o.zip")
    os.system(f"mkdir -p {user_jittor_path}")
    print(user_jittor_path)
    print(data_o_path)
    shutil.unpack_archive(data_o_path, user_jittor_path)
    # TODO: rename the filename??
    # obj_names = []
    # for i in range(2):
    #     fullname = "jittor_nb" + str(i)
    #     md5 = os.popen(f'echo -n {fullname} | md5sum ').read().split()[0]
    #     os.rename(os.path(user_jittor_path, md5), os.path(user_jittor_path, str(i)+".o"))
