Download dataset from https://github.com/Totoro97/NeuS

## train
```
python tools/run_net.py --config-file ./projects/neus/neus_womask.py --type mesh --task train
```

## extract mesh
```
python tools/run_net.py --config-file ./projects/neus/neus_womask.py --type mesh --task validate_mesh
```