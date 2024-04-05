<!-- # JNeRF -->
<div align="center">
<img src="docs/logo.png" height="200"/>
</div>

## Introduction
JNeRF is an NeRF benchmark based on [Jittor](https://github.com/Jittor/jittor). JNeRF supports Instant-NGP capable of training NeRF in 5 seconds and achieves similar performance and speed to the paper.

5s training demo of Instant-NGP implemented by JNeRF:

<img src="docs/demo_5s.gif" width="300"/>

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
* GPU supporting:
  * sm arch >= sm_61 (GTX 10x0 / TITAN Xp and above)
  * to use fp16: sm arch >= sm_70 (TITAN V / V100 and above). JNeRF will automatically use original fp32 if the requirements are not meet.
  * to use FullyFusedMLP: sm arch >= sm_75 (RTX 20x0 and above). JNeRF will automatically use original MLPs if the requirements are not meet.

**Install the requirements & JNeRF**

JNeRF is a benchmark toolkit and can be updated frequently, so installing in editable mode is recommended.
Thus any modifications made to JNeRF will take effect without reinstallation.

```shell
git clone https://github.com/Jittor/JNeRF
cd JNeRF
python3.x -m pip install --user -e .
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

After installation, you can ```import jnerf``` in python interpreter to check if it is successful or not.

## Getting Started

### Datasets

We use fox datasets and blender lego datasets for training demonstrations. 

#### Fox Dataset
We provided fox dataset (from [Instant-NGP](https://github.com/NVlabs/instant-ngp)) in this repository at `./data/fox`.

#### Lego Dataset
You can download the lego dataset in nerf_example_data.zip at https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1. And move `lego` folder to `./data/lego`.

#### Customized Datasets

If you want to train JNerf with your own dataset, then you should follow the format of our datasets. You should split your datasets into training, validation and testing sets. Each set should be paired with a json file that describes the camera parameters of each images.

### Config

We organize our configs of JNeRF in projects/. You are referred to `./projects/ngp/configs/ngp_base.py` to learn how it works.

### Train from scratch

You can train from scratch on the `lego` scene with the following command. It should be noted that since jittor is a just-in-time compilation framework, it will take some time to compile on the first run.
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py
```
NOTE: Competitors participating in the Jittor AI Challenge can use `./projects/ngp/configs/ngp_comp.py` as config.

### Test with pre-trained model

After training, the ckpt file `params.pkl` will be automatically saved in `./logs/lego/`. And you can modify the ckpt file path by setting the `ckpt_path` in the config file. 

Set the `--task` of the command to `test` to test with pre-trained model:
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py --task test
```

### Render demo video

Set the `--task` of the command to `render` to render demo video `demo.mp4` with specified camera path based on pre-trained model:
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py --task render
```

### Extract Mesh with color

After training, you can extract mesh with color from the pre-trained model, and the mesh model file `mesh.ply` will be saved in `./logs/lego/`:
```
python tools/extract_mesh.py --config-file ./projects/ngp/configs/ngp_base.py --resolution 512 --mcube_smooth False
```

### Running other models

Instructions of how to running other models are put in the contrib folder.
```
contrib/
|---mipnerf/
|---pixelnerf/
|---plenoxel/
```

### Running fox dataset

run builtin fox dataset:
```
python3.x ./tools/run_net.py --config ./projects/ngp/configs/ngp_fox.py
```

## Performance

Instant-ngp implemented by JNeRF achieves similar performance and speed to the paper. The performance comparison can be seen in the table below and training speed of JNeRF-NGP on RTX 3090 is about 133 iters/s. 


|    Models     |    implementation      | Dataset | PSNR |
|----|---|---|---|
| Instant-ngp | paper | lego | 36.39(5min) |
| Instant-ngp | JNeRF | lego | 36.41(5min) |
| NeRF        | JNeRF | lego | 32.49 |

## Plan of Models

JNeRF will support more valuable NeRF models in the future, if you are also interested in JNeRF and want to improve it, welcome to submit PR!

<b>:heavy_check_mark:Supported  :clock3:Doing :heavy_plus_sign:TODO</b>

- :heavy_check_mark: Instant-NGP
- :heavy_check_mark: NeRF
- :heavy_check_mark: NeuS
- :heavy_check_mark: Mip-NeRF
- :heavy_check_mark: Plenoxels
- :heavy_check_mark: Recursive-NeRF
- :heavy_check_mark: pixelNeRF
- :clock3: PaletteNeRF
- :heavy_plus_sign: StylizedNeRF
- :heavy_plus_sign: NeRF-Editing
- :heavy_plus_sign: DrawingInStyles
- :heavy_plus_sign: NSVF
- :heavy_plus_sign: NeRFactor
- :heavy_plus_sign: StyleNeRF
- :heavy_plus_sign: EG3D
- :heavy_plus_sign: ...

## Contact Us

If you are interested in JNeRF or NeRF research and want to build the JNeRF community with us, contributing is very welcome, please contact us! 

Email: jittor@qq.com

JNeRF QQ Group: 689063884

<img src="docs/jnerf_qrcode.jpg" width="200"/>

If you have any questions about Jittor, you can [file an issue](https://github.com/Jittor/jittor/issues), or join our Jittor QQ Group: 761222083

<img src="docs/jittor_qrcode.jpg" width="200"/>

## Acknowledgements

The original implementation comes from the following cool project:
* [Instant-NGP](https://github.com/NVlabs/instant-ngp)
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
* [Eigen](https://github.com/Tom94/eigen) ([homepage](https://eigen.tuxfamily.org/index.php?title=Main_Page))

Their licenses can be seen at `licenses/`, many thanks for their nice work!


## Citation


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
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```
