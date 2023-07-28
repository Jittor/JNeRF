# Prepare dataset

Download the dataset from https://github.com/Totoro97/NeuS and extract it.

For example, you can download the example dataset: dtu_scan24, and put it in ./dataset/dtu_scan24

the dataset structure should be: 

```
-- dtu_scan24
    -- images
        -- ...
    -- mask (optional)
        -- ...
    -- cameras_sphere.npz
```


ps. As competition baseline, you should change your `dataset_dir` to the competition dataset path

# Install

You might need to install cupy for Jittor. 

Note that you should install the cupy with correct CUDA version corresponding to your device. 

```
# for CUDA-10.2, run: 
pip install cupy-cuda102
```

# Set the project file

Update path in `JNeRF/projects/neus/configs/neus_womask.py`

```

...

# Line 11-16
dataset = dict(
type = 'NeuSDataset',
# set dataset dir to the path where the dataset is extracted
dataset_dir = 'dataset/dtu_scan24',
render_cameras_name = 'cameras_sphere.npz',
object_cameras_name = 'cameras_sphere.npz',
)

...

# Line 90
# set base_exp_dir to the path where you want to save the results
base_exp_dir = './log/dtu_scan24/womask'

...

```


# Train

Use the following command to train the network.
```
# without mask
python tools/run_net.py --config-file ./projects/neus/configs/neus_womask.py --type mesh --task train
# with mask
python tools/run_net.py --config-file ./projects/neus/configs/neus_wmask.py --type mesh --task train
```

ps. As competition baseline, using `neus_wmask.py` will have better performance.

During training, the network will output `depth`, `normal`, `RGB`, and `mesh`. However, the quality of the mesh is lower at this point to speed up training.

![](./fig/depth.png)
![](./fig/normal.png)
![](./fig/rgb.png)

# Extract mesh

After finishing training the network, you can use the following command to extract the high quality mesh, and the mesh is saved in the results folder ( eg. `./log/dtu_scan24/womask/mesh_512` ).
```
# without mask
python tools/run_net.py --config-file ./projects/neus/configs/neus_womask.py --type mesh --task validate_mesh
# with mask
python tools/run_net.py --config-file ./projects/neus/configs/neus_wmask.py --type mesh --task validate_mesh
```

![](./fig/mesh.png)

# More

For more details, please check https://github.com/Totoro97/NeuS.
