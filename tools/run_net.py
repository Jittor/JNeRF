from ast import parse
import jittor as jt
# from train import Trainer
# from model import NerfNetworks, HuberLoss
from tqdm import tqdm
# from utils.dataset import  NerfDataset
import argparse
import numpy as np
import os
from jnerf.runner import Runner 
from jnerf.utils.config import init_cfg
# jt.flags.gopt_disable=1
jt.flags.use_cuda = 1


def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )

    parser.add_argument(
        "--save_dir",
        default="logs",
        type=str,
    )
    
    args = parser.parse_args()

    assert args.task in ["train","test","render"],f"{args.task} not support, please choose [train, test, render]"
    if args.config_file:
        init_cfg(args.config_file)

    runner = Runner()

    if args.task == "train":
        runner.train()
    elif args.task == "test":
        runner.test()
    elif args.task == "render":
        print("TODO: load model")
        runner.render_all(True, args.save_dir)
    
if __name__ == "__main__":
    main()