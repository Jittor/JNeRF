# JNeRF
## Introduction
JNeRF is an NeRF benchmark based on [Jittor](https://github.com/Jittor/jittor). 

## Install
JNeRF environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

**Step 1: Install the requirements**
```shell
sudo apt-get install tcl-dev tk-dev python3-tk
git clone https://github.com/Jittor/JNeRF
cd JNeRF
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JNeRF**
 
You can add ```export PYTHONPATH=$PYTHONPATH:{you_own_path_to_jittor}/jittor/python:{you_own_path_to_jnerf}/JNeRF/python``` into ```.bashrc```, and run
```shell
source .bashrc
```

## Getting Started

### Datasets

We use fox datasets and blender lego datasets for training. 

Lego: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1. In the blender category.

#### Customized Datasets

If you want to train JNerf with your own dataset, then you should follow the format of our datasets. You should split your datasets into training, validation and testing sets. Each set should be paired with a json file that describes the camera parameters of each images.

### config

We organize our configs of JNeRF in projects/. You are referred to ./projects/ngp/configs/ngp_base.py to learn how it works.

### Train & Test

Train and test are combined in a single command.
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py
```

### Models

|    Models     | Dataset | Image_size | Test PSNR | Optim | Lr schd | Paper | Config  | Download |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| Instant-ngp | lego | 1920/1080 | 36.49 |  Adam+ema  |   expdecay   | [NVIDIA](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)| [config](projects/ngp/configs/ngp_base.py) |  |

## Contact Us


Website: http://cg.cs.tsinghua.edu.cn/jittor/

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

QQ Group: 761222083


<img src="https://cg.cs.tsinghua.edu.cn/jittor/images/news/2020-12-8-21-19-1_2_2/fig4.png" width="200"/>

### The Team


JNeRF is currently maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in JNeRF and want to improve it, Please join us!


### Citation


```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```