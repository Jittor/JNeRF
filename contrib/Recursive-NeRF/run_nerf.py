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

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from tensorboardX import SummaryWriter


jt.flags.use_cuda = 1
# np.random.seed(0)
DEBUG = False

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    # print("x min", inputs_flat[:,0].min(), inputs_flat[:,0].max())
    # print("y min", inputs_flat[:,1].min(), inputs_flat[:,1].max())
    # print("z min", inputs_flat[:,2].min(), inputs_flat[:,2].max())
    embedded = embed_fn(inputs_flat)
    training = not jt.flags.no_grad

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = jt.concat([embedded, embedded_dirs], -1)

    if netchunk is None:
        outputs_flat, loss_conf, loss_num = fn(embedded)
    else:
        flat_list = []
        conf_list = []
        sout_num_list = []
        loss_conf = jt.zeros([1])
        # loss_num = 0
        # pre_value = 0
        for i in range(0, embedded.shape[0], netchunk):
            flat, conf, sout_num = fn(embedded[i:i+netchunk], inputs_flat[i:i+netchunk], training)
            flat_list.append(flat)
            conf_list.append(conf)
            sout_num_list.append(np.expand_dims(sout_num, 0))
            # loss_conf += lconf
            # loss_num += num
            # pre_value = (flat.reshape((-1))[0] + conf.reshape((-1))[0] + lconf.reshape((-1))[0]) * 0
            if not training:
                flat_list[-1].sync()
                conf_list[-1].sync()
                loss_conf.sync()
        outputs_flat = jt.concat(flat_list, 1)
        conf_flat = jt.concat(conf_list, 1)
        sout_flat = np.concatenate(sout_num_list, 0)
        sout_flat = sout_flat.sum(0)
    # loss_conf /= float(loss_num)
    outputs = jt.reshape(outputs_flat, [outputs_flat.shape[0]] + list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    conf = jt.reshape(conf_flat, [conf_flat.shape[0]] + list(inputs.shape[:-1]) + [conf_flat.shape[-1]])
    return outputs, conf, loss_conf, sout_flat


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # jt.display_memory_info()
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
            # print("all_ret[k]",k, len(all_ret[k]), all_ret[k][0].shape)
            # ret[k].sync()
        # jt.display_memory_info()
    for k in all_ret:
        # print("k",k)
        if k=="loss_conf" or k=="loss_conf0":
            all_ret[k] = jt.concat(all_ret[k], 0)
        elif k=="pts":
            all_ret[k] = np.concatenate(all_ret[k], axis=0)
        elif k=="outnum" or k=="outnum0":
            all_ret[k] = np.concatenate(all_ret[k], axis=0).sum(0)
        else:
            all_ret[k] = jt.concat(all_ret[k], 1)
    # all_ret = {k : jt.concat(all_ret[k], 1) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, intrinsic=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        print("render c2w",c2w.shape)
        rays_o, rays_d = get_rays(H, W, focal, c2w, intrinsic)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            assert intrinsic is None
            # special case to visualize effect of viewdirs
            print("render c2w_staticcam",c2w_staticcam.shape)
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / jt.norm(viewdirs, k=2, dim=-1, keepdim=True)
        viewdirs = jt.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = jt.reshape(rays_o, [-1,3]).float()
    rays_d = jt.reshape(rays_d, [-1,3]).float()

    near, far = near * jt.ones_like(rays_d[...,:1]), far * jt.ones_like(rays_d[...,:1])
    rays = jt.concat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = jt.concat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        if k=="loss_conf" or k=="loss_conf0" or k=="pts" or k=="outnum" or k=="outnum0":
            continue
        k_sh = [all_ret[k].shape[0]] + list(sh[:-1]) + list(all_ret[k].shape[2:])
        all_ret[k] = jt.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, intrinsic = None, get_points = False, log_path = None, large_scene = False):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    rgbs_log = []
    disps = []
    outnum0 = []
    outnum = []
    points = None

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # import ipdb
        # ipdb.set_trace()
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], intrinsic=intrinsic, **render_kwargs)
        rgbs.append(rgb[-1].numpy())
        rgbs_log.append(rgb[-2].numpy())
        disps.append(disp[-1].numpy())
        outnum0.append(np.expand_dims(extras['outnum0'], 0))
        outnum.append(np.expand_dims(extras['outnum'], 0))
        if get_points:
            point = extras['pts']
        #     conf = extras['conf_map'][-1].numpy()
        #     point = point.reshape((-1, point.shape[-1]))
        #     conf = conf.reshape((-1))
            # idx=np.random.choice(point.shape[0], point.shape[0]//render_poses.shape[0])
            # point = point[idx]
            if points is None:
                points = point
            else:
                points = np.concatenate((points, point), axis=0)
            
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        del rgb
        del disp
        del acc
        del extras
        
        if large_scene:
            if get_points and intrinsic is None:
                break
        else:
            if get_points:
                break
        if jt.mpi and jt.mpi.local_rank()!=0:
            break

    outnum0 = np.concatenate(outnum0, axis=0).sum(0)
    outnum = np.concatenate(outnum, axis=0).sum(0)
    sout_num = list(outnum0)
    log = ""
    for i in range(len(sout_num)):
        log += str(i)+": %d, " % int(sout_num[i])
    sout_num = list(outnum)
    log += "\n"
    for i in range(len(sout_num)):
        log += str(i)+": %d, " % int(sout_num[i])
    print(log)
    if log_path is not None:
        with open(log_path, 'w') as file_object:
            file_object.write(log)

    rgbs = np.stack(rgbs, 0)
    rgbs_log = np.stack(rgbs_log, 0)
    disps = np.stack(disps, 0)

    if get_points:
        return rgbs, disps, rgbs_log, points
    else:
        return rgbs, disps, rgbs_log

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, head_num=args.head_num, threshold=args.threshold*2.0 if args.large_scene else args.threshold*10.0)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, head_num=args.head_num, threshold=args.threshold)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = jt.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = jt.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'threshold' : args.threshold,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, conf, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=jt.nn.relu: 1.-jt.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = jt.concat([dists, jt.array(np.array([1e10]).astype(np.float32)).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * jt.norm(rays_d.unsqueeze(-2), k=2, dim=-1)

    rgb = jt.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = jt.init.gauss(raw[...,3].shape, raw.dtype) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            # np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = jt.array(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * jt.cumprod(jt.concat([jt.ones((1,alpha.shape[1], 1)), 1.-alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = jt.sum(weights.unsqueeze(-1) * rgb, -2)  # [N_rays, 3]
    # conf_map = jt.sum(weights.unsqueeze(-1) * conf, -2)  # [N_rays, 1]
    # conf_map = jt.mean(conf, -2)  # [N_rays, 1]
    conf_map = conf

    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1./jt.maximum(1e-10 * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))

    return rgb_map, disp_map, acc_map, weights, depth_map, conf_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                threshold=3e-2,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    training = not jt.flags.no_grad
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = jt.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = jt.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = jt.concat([mids, z_vals[...,-1:]], -1)
        lower = jt.concat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = jt.random(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            # np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = jt.array(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw, conf, loss_conf, outnum = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, conf_map = raw2outputs(raw, conf, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, conf_map_0, loss_conf_0, weights_0, outnum_0 = rgb_map, disp_map, acc_map, conf_map, loss_conf, weights, outnum

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[-1,...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        _, z_vals = jt.argsort(jt.concat([z_vals, z_samples], -1), -1)
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw, conf, loss_conf, outnum = network_query_fn(pts, viewdirs, run_fn)
        # print("raw",raw.shape)
        # print("pts",pts.shape)

        rgb_map, disp_map, acc_map, weights, depth_map, conf_map = raw2outputs(raw, conf, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        # print("rgb_map",rgb_map.shape)
        # print("conf_map",conf_map.shape)
    if training:
        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'conf_map' : conf_map, 'loss_conf' : loss_conf, 'weights' : weights}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['conf_map0'] = conf_map_0
            ret['loss_conf0'] = loss_conf_0
            ret['weights0'] = weights_0
            # ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays] TODO: support jt.std
    else:
        point = pts.numpy()
        conf = conf_map[-1].numpy()
        point = point.reshape((-1, point.shape[-1]))
        conf = conf.reshape((-1))
        idx=np.random.choice(point.shape[0], point.shape[0]//10, replace=False)
        point = point[idx]
        conf = conf[idx]
        # threshold = 0.0
        point = point[conf>=threshold]
        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'pts' : point, 'outnum' : np.expand_dims(outnum,0)}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['outnum0'] = np.expand_dims(outnum_0,0)

    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #         print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def dfs(t, points, model, model_fine):
    k = len(model.son_list[t])
    if t in model.force_out:
        if points.shape[0]>=k:
            centroid = points[jt.array(np.random.choice(points.shape[0], k, replace=False))]
            print("centroid",centroid.shape)
            # print("step",-1,centroid.numpy())
            for step in range(100):
                dis = (points.unsqueeze(1) - centroid.unsqueeze(0)).sqr().sum(-1).sqrt()
                min_idx, _ = jt.argmin(dis,-1)
                # print("min_idx",min_idx.shape)
                for i in range(k):
                    # print("i",i,(min_idx==i).sum())
                    centroid[i] = points[min_idx==i].mean(0)
                # print("step",step,centroid.numpy())
        else:
            centroid = jt.rand((k,3))
            print("centroid fail",centroid.shape)
        setattr(model, model.node_list[t].anchors, centroid.detach())
        setattr(model_fine, model_fine.node_list[t].anchors, centroid.detach())
        if jt.mpi:
            # v1 = getattr(model, model.node_list[t].anchors)
            # v1.assign(jt.mpi.broadcast(v2, root=0))
            # v2 = getattr(model_fine, model_fine.node_list[t].anchors)
            # v2.assign(jt.mpi.broadcast(v2, root=0))
            jt.mpi.broadcast(getattr(model, model.node_list[t].anchors), root=0)
            jt.mpi.broadcast(getattr(model_fine, model_fine.node_list[t].anchors), root=0)
        print("model", jt.mpi.local_rank(), t, getattr(model, model.node_list[t].anchors))
        print("modelfine", jt.mpi.local_rank(), t, getattr(model_fine, model_fine.node_list[t].anchors))
        for i in model.son_list[t]:
            model.outnet[i].alpha_linear.load_state_dict(model.outnet[t].alpha_linear.state_dict())
            model_fine.outnet[i].alpha_linear.load_state_dict(model_fine.outnet[t].alpha_linear.state_dict())
            # model.outnet[i].load_state_dict(model.outnet[t].state_dict())
            # model_fine.outnet[i].load_state_dict(model_fine.outnet[t].state_dict())
        return model.son_list[t]
    else:
        centroid = model.get_anchor(model.node_list[t].anchors)
        dis = (points.unsqueeze(1) - centroid.unsqueeze(0)).sqr().sum(-1).sqrt()
        min_idx, _ = jt.argmin(dis,-1)
        res = []
        for i in range(k):
            res += dfs(model.son_list[t][i], points[min_idx==i], model, model_fine)
        return res

def do_kmeans(points, model, model_fine):
    # points = points.reshape(-1, points.shape[-1])
    # confs = confs.reshape(-1)
    print("do_kmeans",points.shape)
    points = jt.array(points)
    force_out = dfs(0, points, model, model_fine)

    model.force_out = force_out
    model_fine.force_out = force_out

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
    parser.add_argument("--step1", type=int, default=5000, 
                        help='?')
    parser.add_argument("--step2", type=int, default=10000, 
                        help='?')
    parser.add_argument("--step3", type=int, default=15000, 
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
    parser.add_argument("--threshold", type=float, default=1e-2,
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
    parser.add_argument("--i_img",     type=int, default=5000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_tottest", type=int, default=400000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    intrinsic = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_test_tot = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        testskip = args.testskip
        faketestskip = args.faketestskip
        if jt.mpi and jt.mpi.local_rank()!=0:
            testskip = faketestskip
            faketestskip = 1
        if args.do_intrinsic:
            images, poses, intrinsic, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, testskip, args.blender_factor, True)
        else:
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, testskip, args.blender_factor)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        i_test_tot = i_test
        i_test = i_test[::faketestskip]

        near = args.near
        far = args.far
        print("near", near)
        print("far", far)

        if args.white_bkgd:
            # accs = images[...,-1]
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

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

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = jt.array(render_poses)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with jt.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, intrinsic = intrinsic)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    accumulation_steps = 2
    N_rand = args.N_rand//accumulation_steps
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = jt.array(images.astype(np.float32))
    # accs = jt.array(accs.astype(np.float32))
    # a=images[0].copy()
    # b=images[1].copy()
    # c=images[2].copy()
    # print("images0",a.sum())
    # print("images1",b.sum())
    # print("images2",c.sum())
    # print("images0",images[0].numpy().sum())
    # print("images1",images[1].numpy().sum())
    # print("images2",images[2].numpy().sum())
    # print("images0",images[0].sum().numpy())
    # print("images1",images[1].sum().numpy())
    # print("images2",images[2].sum().numpy())
    poses = jt.array(poses)
    if use_batching:
        rays_rgb = jt.array(rays_rgb)


    N_iters = 300000*accumulation_steps + 1
    split_tree1 = args.step1*accumulation_steps
    split_tree2 = args.step2*accumulation_steps
    split_tree3 = args.step3*accumulation_steps
    # split_tree1 = args.i_img
    # split_tree2 = args.i_img*2
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    if not jt.mpi or jt.mpi.local_rank()==0:
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                        .replace(":", "")\
                                        .replace(" ", "_")
        gpu_idx = os.environ["CUDA_VISIBLE_DEVICES"]
        log_dir = os.path.join("./logs", "summaries", f"log_{date}_gpu{gpu_idx}_{args.expname}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    start = start + 1
    # if not use_batching and jt.mpi:
    #     img_i_list = np.random.choice(i_train, N_iters)
    #     print("before img_i_list",img_i_list.sum())
    #     jt.mpi.broadcast(img_i_list, root=0)
    #     print("after img_i_list",img_i_list.sum())
    for i in trange(start, N_iters):
        # print("i",i,"jt.mpi.local_rank()",jt.mpi.local_rank())
        # jt.display_memory_info()
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = jt.transpose(batch, (1, 0, 2))
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = jt.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            # if jt.mpi:
            #     img_i = img_i_list[i]
            # else:
            #     img_i = np.random.choice(i_train)
            img_i = np.random.choice(i_train)
            target = images[img_i]#.squeeze(0)
            # acc_target = accs[img_i]
            pose = poses[img_i, :3,:4]#.squeeze(0)

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose, intrinsic)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = jt.stack(
                        jt.meshgrid(
                            jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = jt.reshape(coords, [-1,2])  # (H * W, 2)
                # if jt.mpi:
                #     assert coords.shape[0]%jt.mpi.world_size()==0
                #     select_inds = np.random.choice(coords.shape[0]//jt.mpi.world_size(), size=[N_rand], replace=False)  # (N_rand,)
                #     select_inds += (coords.shape[0]//jt.mpi.world_size())*jt.mpi.local_rank()
                # else:
                #     select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].int()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = jt.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                # target_a = acc_target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        # print("rgb",rgb.shape)
        # print("target_s",target_s.shape)
        img_loss = img2mse(rgb, target_s.unsqueeze(0))
        # img_loss = (img2mse(rgb[-1], target_s.unsqueeze(0))*10.0+img2mse(rgb[:-1], target_s.unsqueeze(0)))/11.0
        trans = extras['raw'][...,-1]
        loss = img_loss
        # loss_conf = img2mse(extras['conf_map'], (rgb-target_s)**2)
        # loss_conf = jt.maximum((rgb-target_s)**2-extras['conf_map'],0.)+jt.maximum(extras['conf_map']+0.01,0.).mean()
        sloss = jt.maximum(((rgb-target_s)**2).detach().unsqueeze(-2)-extras['conf_map'],0.)
        # loss_conf_loss = jt.sum(extras['weights'].unsqueeze(-1) * sloss, -2).mean()
        loss_conf_loss = sloss.mean()
        # loss_conf_loss = jt.maximum((rgb-target_s)**2-extras['conf_map'],0.).mean()
        loss_conf_mean = jt.maximum(extras['conf_map'], 0.).mean()
        # print("(rgb-target_s)",(rgb-target_s).shape)
        # print("loss_conf_loss",loss_conf_loss.shape)
        # print("loss_conf_mean",loss_conf_mean.shape)
        # loss_conf = jt.maximum((rgb-target_s)**2-extras['conf_map'],0.).sum()
        # print("ssqr",ssqr.shape,"mean",ssqr.mean(),"max",ssqr.max())
        # print("extras['conf_map']",extras['conf_map'].shape,"mean",extras['conf_map'].mean(),"max",extras['conf_map'].max())
        # print("extras['loss_conf']",extras['loss_conf'].shape,"mean",extras['loss_conf'].mean(),"max",extras['loss_conf'].max())
        # psnr = mse2psnr(img_loss)
        # psnr = mse2psnr(img2mse(rgb[-2], target_s.unsqueeze(0)))
        psnr = mse2psnr(img2mse(rgb[-1], target_s.unsqueeze(0)))
        # loss_acc = jt.zeros([1])
        # if args.white_bkgd:
        #     # print("acc",acc.shape)
        #     # print("target_a",target_a.shape)
        #     loss_acc = loss_acc+img2mse(acc, target_a.unsqueeze(0))

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            # img_loss0 = (img2mse(extras['rgb0'][-1], target_s.unsqueeze(0))*10.0+img2mse(extras['rgb0'][:-1], target_s.unsqueeze(0)))/11.0
            loss = loss + img_loss0
            # loss_conf = loss_conf + img2mse(extras['conf_map0'], (extras['rgb0']-target_s)**2)
            # loss_conf = loss_conf + jt.maximum((extras['rgb0']-target_s)**2-extras['conf_map0'],0.)+jt.maximum(extras['conf_map0']+0.01,0.)
            sloss = jt.maximum(((extras['rgb0']-target_s)**2).detach().unsqueeze(-2)-extras['conf_map0'],0.)
            # loss_conf_loss = loss_conf_loss + jt.sum(extras['weights0'].unsqueeze(-1) * sloss, -2).mean()
            loss_conf_loss = loss_conf_loss + sloss.mean()
            # loss_conf_loss = loss_conf_loss + jt.maximum((extras['rgb0']-target_s)**2-extras['conf_map0'],0.).mean()
            loss_conf_mean = loss_conf_mean + jt.maximum(extras['conf_map0'], 0.).mean()
            # loss_conf = loss_conf + jt.maximum((extras['rgb0']-target_s)**2-extras['conf_map0'],0.).sum()
            psnr0 = mse2psnr(img_loss0)
            # if args.white_bkgd:
            #     loss_acc = loss_acc+img2mse(extras['acc0'], target_a.unsqueeze(0))
        loss_conf_loss = loss_conf_loss
        # print("loss_conf_loss",loss_conf_loss)
        # print("loss_conf_mean",loss_conf_mean)
        loss_conf = loss_conf_loss+loss_conf_mean*0.01
        loss = loss + loss_conf*0.1
        # loss = loss + loss_acc

        jt.sync_all()
        optimizer.backward(loss / accumulation_steps)
        if i % accumulation_steps == 0:
            optimizer.step()
        jt.sync_all()
        # optimizer.step(loss)

        # if global_step==10000:
        # if global_step==0:
        #     render_kwargs_train['network_fn'].force_out = 15
        #     render_kwargs_train['network_fine'].force_out = 15

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * accumulation_steps * 1000
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
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if (i+1)%args.i_weights==0:
            if (not jt.mpi or jt.mpi.local_rank()==0):
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            else:
                path = os.path.join(basedir, expname, 'tmp.tar')
            jt.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # jt.display_memory_info()
        if i%args.i_video==0 and i > 0 or i==split_tree1 or i==split_tree2 or i==split_tree3:
                # import ipdb 
                # ipdb.set_trace()
                # Turn on testing mode
                if not jt.mpi or jt.mpi.local_rank()==0:
                    with jt.no_grad():
                        rgbs, disps, rgbs_log, points = render_path(render_poses, hwf, args.chunk, render_kwargs_test, intrinsic = intrinsic, get_points = True, large_scene = args.large_scene)
                else:
                    points = jt.random([1000,3])
                if i==split_tree1 or i==split_tree2 or i==split_tree3:
                    do_kmeans(points, render_kwargs_train['network_fn'], render_kwargs_train['network_fine'])
                # jt.display_memory_info()
                if not jt.mpi or jt.mpi.local_rank()==0:
                    print('Done, saving', rgbs.shape, disps.shape)
                    moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                    imageio.mimwrite(moviebase + 'rgb_log.mp4', to8b(rgbs_log), fps=30, quality=8)
                    imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
                jt.gc()

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with jt.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)
        if i%args.i_testset==0 and i > 0 and (not jt.mpi or jt.mpi.local_rank()==0):
            si_test = i_test_tot if i%args.i_tottest==0 else i_test
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[si_test].shape)
            with jt.no_grad():
                rgbs, disps, rgbs_log = render_path(jt.array(poses[si_test]), hwf, args.chunk, render_kwargs_test, gt_imgs=images[si_test], savedir=testsavedir, intrinsic = intrinsic, log_path = os.path.join(basedir, expname, 'outnum_{:06d}.txt'.format(i)))
            tars = images[si_test]
            testpsnr = mse2psnr(img2mse(jt.array(rgbs), tars)).item()
            if not jt.mpi or jt.mpi.local_rank()==0:
                writer.add_scalar('test/psnr_tot', testpsnr, global_step)
                print('Saved test set')



    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} expname: {args.expname} Loss: {loss.item()}  LossConf: {loss_conf.item()}  PSNR: {psnr.item()}")
            a=psnr0.item()
            # print("before:",jt.mpi.local_rank())
            if not jt.mpi or jt.mpi.local_rank()==0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                # writer.add_scalar("train/loss_acc", loss_acc.item(), global_step)
                writer.add_scalar("train/loss_conf", loss_conf.item(), global_step)
                writer.add_scalar("train/PSNR", psnr.item(), global_step)
                writer.add_scalar("lr/lr", new_lrate, global_step)
                # writer.add_histogram('tran', trans.numpy(), global_step)
                if args.N_importance > 0:
                    writer.add_scalar("train/PSNR0", psnr0.item(), global_step)
            # print("done:",jt.mpi.local_rank())
            # print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            # print('iter time {:.05f}'.format(dt))

            # with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
            #     tf.contrib.summary.scalar('loss', loss)
            #     tf.contrib.summary.scalar('psnr', psnr)
            #     tf.contrib.summary.histogram('tran', trans)
            #     if args.N_importance > 0:
            #         tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with jt.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, intrinsic=intrinsic,
                                                        **render_kwargs_test)
                sout_num = list(extras['outnum0'])
                log = "i_img out_num0:"
                for k in range(len(sout_num)):
                    log += str(k)+": %d;  " % int(sout_num[k])
                print(log)
                sout_num = list(extras['outnum'])
                log = "i_img out_num:"
                for k in range(len(sout_num)):
                    log += str(k)+": %d;  " % int(sout_num[k])
                print(log)
                psnr = mse2psnr(img2mse(rgb[-1], target))
                # psnr = mse2psnr(img2mse(rgb[-2], target))
                rgb_log = rgb[-2].numpy()
                rgb = rgb[-1].numpy()
                rgb0 = extras['rgb0'][-1].numpy()
                # rgb = extras['rgb0'][-1].numpy()
                disp = disp[-1].numpy()
                acc = acc[-1].numpy()
                acc0 = extras['acc0'][-1].numpy()
                a = target.numpy()
                a = psnr.item()

                if not jt.mpi or jt.mpi.local_rank()==0:
                    writer.add_image('test/rgb', to8b(rgb), global_step, dataformats="HWC")
                    writer.add_image('test/rgb0', to8b(rgb0), global_step, dataformats="HWC")
                    writer.add_image('log/rgb', to8b(rgb_log), global_step, dataformats="HWC")
                    # writer.add_image('test/disp', disp[...,np.newaxis], global_step, dataformats="HWC")
                    # writer.add_image('test/acc', acc[...,np.newaxis], global_step, dataformats="HWC")
                    # writer.add_image('test/acc0', acc0[...,np.newaxis], global_step, dataformats="HWC")
                    writer.add_image('test/target', target.numpy(), global_step, dataformats="HWC")

                    writer.add_scalar('test/psnr', psnr.item(), global_step)
                jt.gc()
                # jt.display_memory_info()
                # jt.display_max_memory_info()
                # a=images[0].numpy()
                # b=images[1].numpy()
                # c=images[2].numpy()
                # print("images0",a.shape,a.sum())
                # print("images1",b.shape,b.sum())
                # print("images2",c.shape,c.sum())
                # writer.add_image('test/rgb_target0', a, global_step, dataformats="HWC")
                # writer.add_image('test/rgb_target1', b, global_step, dataformats="HWC")
                # writer.add_image('test/rgb_target2', c, global_step, dataformats="HWC")


                # if args.N_importance > 0:

                #     with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                #         tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                #         tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                #         tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        

        global_step += 1


if __name__=='__main__':
    train()
