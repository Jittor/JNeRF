import os, sys
import numpy as np
import imageio
import json
import random
import time
import jittor as jt
from jittor import nn
from tqdm import tqdm, trange
import datetime
import cv2

import matplotlib.pyplot as plt

from image_mem_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from tensorboardX import SummaryWriter

jt.flags.use_cuda = 1
# np.random.seed(0)
DEBUG = False

def config_parser():
    gpu = "gpu"+os.environ["CUDA_VISIBLE_DEVICES"]
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/'+gpu+"/", 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--step1", type=int, default=10000, 
                        help='?')
    parser.add_argument("--step2", type=int, default=20000, 
                        help='?')
    parser.add_argument("--step3", type=int, default=3000000, 
                        help='?')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*8,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--threshold", type=float, default=0,
                        help='threshold')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--head_num", type=int, default=8,
                        help='number of heads')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--large_scene", action='store_true', 
                        help='use large scene')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--faketestskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--near", type=float, default=2., 
                        help='downsample factor for LLFF images')
    parser.add_argument("--far", type=float, default=6., 
                        help='downsample factor for LLFF images')
    parser.add_argument("--do_intrinsic", action='store_true', 
                        help='use intrinsic matrix')
    parser.add_argument("--blender_factor", type=int, default=1, 
                        help='downsample factor for blender images')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=50000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_tottest", type=int, default=400000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    parser.add_argument("--scaledown",   type=int, default=1, 
                        help='frequency of render_poses video saving')

    return parser

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, 2)
    input_ch_views = 0
    output_ch = 3
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, 
                 head_num=args.head_num, threshold=args.threshold)
    optimizer = jt.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    
    return model, optimizer, embed_fn
    
def render(H, W, x, chunk, embed_fn, model):
    rgb = []
    confs = []
    outnums = []
    for i in range(0,x.shape[0],chunk):
        xx = x[i:i+chunk]
        feat = embed_fn(x[i:i+chunk])
        y, conf, sout_num = model(feat, xx, False)
        rgb.append(y[-1])
        confs.append(conf[-1])
        outnums.append(sout_num)
        rgb[-1].sync()
        confs[-1].sync()
    rgb = jt.concat(rgb, 0).reshape(H,W,3)
    confs = jt.concat(confs, 0)
    outnum = np.concatenate(outnums, axis=0)
    print("outnum",outnum.shape)
    outnum = outnum.sum(0)
    print("outnum",outnum.shape,outnum)
    return rgb, confs

def dfs(t, points, model):
    k = len(model.son_list[t])
    print("dfs",t,"k",k)
    print("points",points.shape)
    if t in model.force_out:
        if points.shape[0]>=k:
            centroid = points[jt.array(np.random.choice(points.shape[0], k, replace=False))]
            print("centroid",centroid.shape)
            # print("step",-1,centroid.numpy())
            for step in range(100):
                # print("step",step)
                dis = (points.unsqueeze(1) - centroid.unsqueeze(0)).sqr().sum(-1).sqrt()
                min_idx, _ = jt.argmin(dis,-1)
                # print("min_idx",min_idx.shape)
                for i in range(k):
                    # print("i",i,(min_idx==i).sum)
                    centroid[i] = points[min_idx==i].mean(0)
                jt.sync_all()
                # jt.display_memory_info()z
                # print("step",step,centroid.numpy())
        else:
            centroid = jt.rand((k,2))
            print("centroid fail",centroid.shape)
        print("centroid",centroid.shape,centroid)
        setattr(model, model.node_list[t].anchors, centroid.detach())
        # warning mpi
        if jt.mpi and False:
            # v1 = getattr(model, model.node_list[t].anchors)
            # v1.assign(jt.mpi.broadcast(v2, root=0))
            jt.mpi.broadcast(getattr(model, model.node_list[t].anchors), root=0)
        print("model", jt.mpi.local_rank(), t, getattr(model, model.node_list[t].anchors))
        for i in model.son_list[t]:
            # model.outnet[i].alpha_linear.load_state_dict(model.outnet[t].alpha_linear.state_dict())
            model.outnet[i].load_state_dict(model.outnet[t].state_dict())
        return model.son_list[t]
    else:
        centroid = model.get_anchor(model.node_list[t].anchors)
        dis = (points.unsqueeze(1) - centroid.unsqueeze(0)).sqr().sum(-1).sqrt()
        min_idx, _ = jt.argmin(dis,-1)
        res = []
        for i in range(k):
            res += dfs(model.son_list[t][i], points[min_idx==i], model)
        return res
        
def do_kmeans(pts, model):
    force_out = dfs(0, pts, model)
    model.force_out = force_out

def print_conf_img(rgb, conf, threshold, testsavedir):
    os.makedirs(testsavedir, exist_ok=True)
    # threshold_list = [threshold, 2e-2, 3e-2, 4e-2, 5e-2, 1e-1, 5e-1, 1]
    threshold_list = [threshold]
    H, W, C = rgb.shape
    red = np.array([1,0,0])
    for th in threshold_list:
        filename = os.path.join(testsavedir, 'confimg_'+str(th)+'.png')
        logimg = rgb.copy().reshape([-1,3])
        print("logimg1",logimg.shape)
        # bo = (conf<th).repeat(1,3)
        bo = (conf<th)[:,0]
        print("bo",bo.shape)
        print("logimg[bo]",logimg[bo].shape)
        logimg[bo] = red
        print("logimg2",logimg.shape)
        logimg=logimg.reshape([H,W,C])
        print("logimg3",logimg.shape)
        imageio.imwrite(filename, to8b(logimg))

def train():
    parser = config_parser()
    args = parser.parse_args()
    
    image = cv2.imread(args.datadir)
    H, W, C = image.shape
    scaledown = args.scaledown
    H = H//scaledown
    W = W//scaledown
    image=cv2.resize(image, (H,W))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = (image / 255.).astype(np.float32)
    image = jt.array(image)
    print("image",image.shape)
    H, W, C = image.shape
    N_iters = 200000 + 1
    split_tree1 = args.step1
    split_tree2 = args.step2
    split_tree3 = args.step3

    coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)
    coords = jt.reshape(coords, [-1,2])  # (H * W, 2)
    coords_f = coords.clone()  # (H * W, 2)
    coords_f[:,0] /= H
    coords_f[:,1] /= W

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    print("use_batching",use_batching)
    if use_batching:
        rand_idx = jt.randperm(coords_f.shape[0])
        rand_coords = coords_f[rand_idx]
        rand_image = image.reshape([-1,3])[rand_idx]
        i_batch = 0
    
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                    .replace(":", "")\
                                    .replace(" ", "_")
    gpu_idx = os.environ["CUDA_VISIBLE_DEVICES"]
    log_dir = os.path.join("./logs", "summaries", f"log_{date}_gpu{gpu_idx}_{args.expname}")
    writer = SummaryWriter(log_dir=log_dir)

    model, optimizer, embed_fn = create_nerf(args)

    start = 1
    global_step = start
    for i in trange(start, N_iters):
        if use_batching:
            x = rand_coords[i_batch:i_batch+N_rand]
            target = rand_image[i_batch:i_batch+N_rand]
            i_batch += N_rand
            if i_batch >= rand_coords.shape[0]:
                # print("Shuffle data after an epoch!")
                rand_idx = jt.randperm(coords_f.shape[0])
                rand_coords = coords_f[rand_idx]
                rand_image = image.reshape([-1,3])[rand_idx]
                i_batch = 0
        else:
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            x = coords_f[select_inds]  # (N_rand, 2)
            target = image[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            # print("select_coords",select_coords.shape,select_coords)
            # print("x",x.shape,x)
            # print("target",target.shape,target)
        
        # if i == 100:
        #     import cProfile
        #     pr = cProfile.Profile()
        #     pr.enable()
        #     jt.flags.trace_py_var=3
        #     jt.flags.profiler_enable = 1
        # elif i == 110:
        #     pr.disable()
        #     pr.print_stats()
        #     jt.flags.profiler_enable = 0
        #     jt.profiler.report()
        #     jt.flags.trace_py_var=0
        feat = embed_fn(x)
        y, conf, _ = model(feat, x, True)
        save=((y-target)**2).detach().unsqueeze(-2)
        # loss = img2mse(y[-1], target)
        loss = img2mse(y, target.unsqueeze(0))
        psnr = mse2psnr(loss)
        
        sloss = jt.maximum(((y-target)**2).detach()-conf,0.)
        loss_conf_loss = sloss.mean()
        loss_conf_mean = jt.maximum(conf, 0.).mean()
        loss_conf = loss_conf_loss+loss_conf_mean*0.01
        loss = loss + loss_conf*0.1
        
        # optimizer.backward(loss)
        optimizer.step(loss)
        jt.sync_all()
        # jt.display_memory_info()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        sstep = global_step
        if sstep>split_tree3:
            sstep-=split_tree3
        elif sstep>split_tree2:
            sstep-=split_tree2
        elif sstep>split_tree1:
            sstep-=split_tree1
        new_lrate = args.lrate * (decay_rate ** (sstep / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/loss_conf", loss_conf.item(), global_step)
            writer.add_scalar("train/PSNR", psnr.item(), global_step)
            writer.add_scalar("lr/lr", new_lrate, global_step)
        if i%args.i_testset==0:
            with jt.no_grad():
                rgb, _ = render(H, W, coords_f, N_rand, embed_fn, model)
                print("rgb",rgb.shape)
                psnr = mse2psnr(img2mse(rgb, image))
            rgb = rgb.numpy()
            
            writer.add_scalar('test/psnr', psnr.item(), global_step)
            if i%args.i_img==0:
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                imageio.imwrite(filename, to8b(rgb))

                writer.add_image('test/rgb', to8b(rgb), global_step, dataformats="HWC")
                writer.add_image('test/target', image.numpy(), global_step, dataformats="HWC")
        if i==split_tree1 or i==split_tree2 or i==split_tree3:
            with jt.no_grad():
                rgb, conf = render(H, W, coords_f, N_rand, embed_fn, model)
                print("split conf",conf.shape)
                print("split coords_f",coords_f.shape)
                pts = coords_f[conf.squeeze(1)>=args.threshold]
                print("split pts threshold:",pts.shape)
                # pts = coords_f[conf>=2e-2]
                # print("split pts 2e-2:",pts.shape)
                # pts = coords_f[conf>=3e-2]
                # print("split pts 3e-2:",pts.shape)
                # pts = coords_f[conf>=4e-2]
                # print("split pts 4e-2:",pts.shape)
                # pts = coords_f[conf>=5e-2]
                # print("split pts 5e-2:",pts.shape)
                # pts = coords_f[conf>=1e-1]
                # print("split pts 1e-1:",pts.shape)
                # pts = coords_f[conf>=5e-1]
                # print("split pts 5e-2:",pts.shape)
                # pts = coords_f[conf>=1]
                # print("split pts 1e-0:",pts.shape)
                do_kmeans(pts, model)
                # print_conf_img(rgb.numpy(), conf.numpy(), args.threshold, os.path.join(basedir, expname, 'confimg_{:06d}'.format(i)))
            jt.gc()
        global_step += 1


if __name__=='__main__':
    train()
