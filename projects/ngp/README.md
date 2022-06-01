# Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

## Getting Started

Train and test on `lego` scene are combined in a single command. It should be noted that since jittor is a just-in-time compilation framework, it will take some time to compile on the first run.
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py
```

## Performance

JNeRF-NGP(Instant-ngp implemented by JNeRF) achieves similar performance and speed to the paper. The performance comparison can be seen in the table below and training speed of JNeRF-NGP on RTX 3090 is about 133 iters/s. 


|    Models     |    implementation      | Dataset | PSNR |
|----|---|---|---|
| Instant-ngp | paper | lego | 36.39 (5min) |
| Instant-ngp | JNeRF | lego | 36.41 (5min) |