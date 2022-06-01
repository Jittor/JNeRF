# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

## Getting Started

Train and test on `lego` scene are combined in a single command. It should be noted that since jittor is a just-in-time compilation framework, it will take some time to compile on the first run.
```shell
python tools/run_net.py --config-file ./projects/nerf/configs/nerf_base.py
```

## Performance


|    Models     |    implementation      | Dataset | PSNR |
|----|---|---|---|
| NeRF | JNeRF | lego | 32.54 |

## TODO
* Support coarse-to-fine sampler
* Compare with the performance of the official implementation
