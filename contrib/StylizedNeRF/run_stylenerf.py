import time
import shutil
import VGGNet
from rendering import *

import jittor as jt
from jittor import nn

import torch.nn as torch_nn
from dataset import RaySampler, StyleRaySampler, StyleRaySampler_gen, LightDataLoader
from models_jt import StyleNerf, StyleMLP_Wild_multilayers, VAE, StyleLatents_variational
from train_style_modules import train_temporal_invoke, train_temporal_invoke_pl
from config import config_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

jt.flags.use_cuda = 1


def train(args):

    """Check Nerf Type"""
    nerf_dict = {'style_nerf': StyleNerf}
    nerf_type_str = ''
    for nerf_type in nerf_dict.keys():
        nerf_type_str += (nerf_type + ' ')
    assert args.nerf_type in nerf_dict.keys(), 'Unknown nerf type: ' + args.nerf_type + '. Only support: ' + nerf_type_str
    print('Type of nerf: ', args.nerf_type)

    """Style Module Type"""
    style_module_dict = {'mlp': StyleMLP_Wild_multilayers}
    style_type_str = ''
    for style_type in style_module_dict.keys():
        style_type_str += (style_type + ' ')
    assert args.style_type in style_module_dict.keys(), 'Unknown style type: ' + args.style_type + '. Only support: ' + style_type_str
    print('Type of style: ', args.style_type)

    """Latent Module Type"""
    latent_module_dict = {'variational': StyleLatents_variational}
    latent_type_str = ''
    for latent_type in latent_module_dict.keys():
        latent_type_str += (latent_type + ' ')
    assert args.latent_type in latent_module_dict.keys(), 'Unknown latent type: ' + args.latent_type + '. Only support: ' + latent_type_str
    print('Type of latent: ', args.latent_type)

    """Check Sampling Type"""
    samp_dict = {'uniform': sampling_pts_uniform}
    samp_type_str = ''
    for samp_type in samp_dict.keys():
        samp_type_str += (samp_type + ' ')
    assert args.sample_type in samp_dict.keys(), 'Unknown nerf type: ' + args.sample_type + '. Only support: ' + samp_type_str
    print('Sampling Strategy: ', args.sample_type)
    samp_func = samp_dict[args.sample_type]
    if args.N_samples_fine > 0:
        samp_func_fine = sampling_pts_fine_jt

    """Saving Configuration"""
    use_viewdir_str = '_UseViewDir_' if args.use_viewdir else ''
    sv_path = os.path.join(args.basedir, args.expname + '_' + args.nerf_type + '_' + args.act_type + use_viewdir_str + 'ImgFactor' + str(int(args.factor)))
    save_makedir(sv_path)
    shutil.copy(args.config, sv_path)
    nerf_gen_data_path = sv_path + '/nerf_gen_data2/'

    """Create Nerfs"""
    nerf = nerf_dict[args.nerf_type]
    model = nerf(args=args, mode='coarse')
    model.train()
    grad_vars = list(model.parameters())
    model_forward = batchify(lambda **kwargs: model(**kwargs), args.chunk)
    if args.N_samples_fine > 0:
        nerf_fine = nerf_dict[args.nerf_type_fine]
        model_fine = nerf_fine(args=args, mode='fine')
        model_fine.train()
        grad_vars += list(model_fine.parameters())
        model_forward_fine = batchify(lambda **kwargs: model_fine(**kwargs), args.chunk)
    optimizer = nn.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    """Create Style Module"""
    style = style_module_dict[args.style_type]
    style_model = style(args)
    style_model.train()
    style_vars = style_model.parameters()
    style_forward = batchify(lambda **kwargs: style_model(**kwargs), args.chunk)
    style_optimizer = nn.Adam(params=style_vars, lr=args.lrate, betas=(0.9, 0.999))

    """VGG and Decoder"""
    decoder = VGGNet.decoder
    vgg = VGGNet.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load('./pretrained/decoder.pth'))
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    vgg = torch_nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    decoder.to(device)

    """Load Check Point"""
    global_step = 0
    ckpts_path = sv_path
    save_makedir(ckpts_path)
    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' not in f and 'latent' not in f]
    print('Found ckpts', ckpts, ' from ', ckpts_path)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading Nerf Model from ', ckpt_path)
        ckpt = jt.load(ckpt_path)
        global_step = ckpt['global_step']
        # Load model
        model.load_state_dict(ckpt['model'])
        # Load optimizer
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.N_samples_fine > 0:
            model_fine.load_state_dict((ckpt['model_fine']))
    ckpts_style = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' in f and 'latent' not in f]
    if len(ckpts_style) > 0 and not args.no_reload:
        ckpt_path_style = ckpts_style[-1]
        print('Reloading Style Model from ', ckpt_path_style)
        ckpt_style = jt.load(ckpt_path_style)
        global_step = ckpt_style['global_step']
        style_model.load_state_dict(ckpt_style['model'])
        style_optimizer.load_state_dict(ckpt_style['optimizer'])

    def Prepare_Style_data(nerf_gen_data_path):
        """Dataset Creation"""
        tmp_dataset = StyleRaySampler(data_path=args.datadir, style_path=args.styledir, factor=args.factor,
                                      mode='valid', valid_factor=args.gen_factor, dataset_type=args.dataset_type,
                                      white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc,
                                      pixel_alignment=args.pixel_alignment, spherify=args.spherify, TT_far=args.TT_far)
        tmp_dataloader = DataLoader(tmp_dataset, args.batch_size_style, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=(args.num_workers > 0))
        print("Preparing nerf data for style training ...")
        cal_geometry(model_forward=model_forward, samp_func=samp_func, dataloader=tmp_dataloader, args=args,
                     device=device,
                     sv_path=nerf_gen_data_path, model_forward_fine=model_forward_fine,
                     samp_func_fine=samp_func_fine)

    """Train 2D Style"""
    if not global_step + 1 < args.origin_step:
        sv_name = '/decoder.pth'
        is_ndc = (args.dataset_type == 'llff' and not args.no_ndc)
        if not os.path.exists(sv_path + sv_name):
            if not os.path.exists(nerf_gen_data_path):
                Prepare_Style_data(nerf_gen_data_path=nerf_gen_data_path)
            print('Training 2D Style Module')
            if args.dataset_type == 'llff':
                train_temporal_invoke(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/', is_ndc=is_ndc,
                                      nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)
            else:
                train_temporal_invoke_pl(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
                                         nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)

    """Dataset Creation"""
    if global_step + 1 < args.origin_step and not os.path.exists(nerf_gen_data_path):
        train_dataset = RaySampler(data_path=args.datadir, factor=args.factor,
                                   mode='train', valid_factor=args.valid_factor, dataset_type=args.dataset_type,
                                   white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc,
                                   pixel_alignment=args.pixel_alignment, spherify=args.spherify, TT_far=args.TT_far)
    else:

        if not os.path.exists(nerf_gen_data_path):
            Prepare_Style_data(nerf_gen_data_path=nerf_gen_data_path)
        train_dataset = StyleRaySampler_gen(data_path=args.datadir, gen_path=nerf_gen_data_path, style_path=args.styledir,
                                            factor=args.factor,
                                            mode='train', valid_factor=args.valid_factor, dataset_type=args.dataset_type,
                                            white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc,
                                            pixel_alignment=args.pixel_alignment, spherify=args.spherify,
                                            decode_path=sv_path+'/decoder.pth',
                                            store_rays=args.store_rays, TT_far=args.TT_far)
        """VAE"""
        vae = VAE(data_dim=1024, latent_dim=args.vae_latent, W=args.vae_w, D=args.vae_d,
                  kl_lambda=args.vae_kl_lambda)
        vae.eval()
        vae_ckpt = args.vae_pth_path
        vae.load_state_dict(torch.load(vae_ckpt))

        """Latents Module"""
        latent_model_class = latent_module_dict[args.latent_type]
        latents_model = latent_model_class(style_num=train_dataset.style_num, frame_num=train_dataset.frame_num, latent_dim=args.vae_latent)
        vae.to(device)
        latent_ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' not in f and 'latent' in f]
        print('Found ckpts', latent_ckpts, ' from ', ckpts_path, ' For Latents Module.')
        if len(latent_ckpts) > 0 and not args.no_reload:
            latent_ckpt_path = latent_ckpts[-1]
            print('Reloading Latent Model from ', latent_ckpt_path)
            latent_ckpt = jt.load(latent_ckpt_path)
            latents_model.load_state_dict(latent_ckpt['train_set'])
        else:
            vae.to(device)
            print("Initializing Latent Model")
            # Calculate and Initialize Style Latents
            all_style_features = torch.from_numpy(train_dataset.style_features).float().to(device)
            _, style_latents_mu, style_latents_logvar = vae.encode(all_style_features)
            # all_style_latents = all_style_latents_mu
            # Set Latents
            latents_model.style_latents_mu = jt.Var(style_latents_mu.detach().cpu().numpy())
            latents_model.style_latents_logvar = jt.Var(style_latents_logvar.detach().cpu().numpy())
            latents_model.set_latents()
        latents_model
        vae.cpu()

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=(args.num_workers > 0))

    # Render valid
    if args.render_valid:
        render_path = os.path.join(sv_path, 'render_valid_' + str(global_step))
        valid_dataset = train_dataset
        valid_dataset.mode = 'valid'
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
        with torch.no_grad():
            if args.N_samples_fine > 0:
                rgb_map, t_map, rgb_map_fine, t_map_fine = render(model_forward=model_forward, samp_func=samp_func, dataloader=valid_dataloader,
                                                                  args=args, device=device, sv_path=render_path, model_forward_fine=model_forward_fine,
                                                                  samp_func_fine=samp_func_fine)
            else:
                rgb_map, t_map, _, _ = render(model_forward=model_forward, samp_func=samp_func, dataloader=valid_dataloader,
                                              args=args, device=device, sv_path=render_path)
        print('Done, saving', rgb_map.shape, t_map.shape)
        exit(0)

    # Render train
    if args.render_train:
        render_path = os.path.join(sv_path, 'render_train_' + str(global_step))
        render_dataset = train_dataset
        if args.N_samples_fine > 0:
            render_train(samp_func=samp_func, model_forward=model_forward, dataset=render_dataset, args=args, device=device, sv_path=render_path, model_forward_fine=model_forward_fine, samp_func_fine=samp_func_fine)
        else:
            render_train(samp_func=samp_func, model_forward=model_forward, dataset=render_dataset, args=args, device=device, sv_path=render_path)
        exit(0)

    # Render valid style
    if args.render_valid_style:
        render_path = os.path.join(sv_path, 'render_valid_' + str(global_step))
        # Enable style
        model.set_enable_style(True)
        if args.N_samples_fine > 0:
            model_fine.set_enable_style(True)
        valid_dataset = train_dataset
        valid_dataset.mode = 'valid_style'
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
        with torch.no_grad():
            if args.N_samples_fine > 0:
                rgb_map, t_map, rgb_map_fine, t_map_fine = render_style(model_forward=model_forward, samp_func=samp_func, style_forward=style_forward, latents_model=latents_model,
                                                                        dataloader=valid_dataloader, args=args, device=device, sv_path=render_path,
                                                                        model_forward_fine=model_forward_fine, samp_func_fine=samp_func_fine, sigma_scale=args.sigma_scale)
            else:
                rgb_map, t_map, _, _ = render_style(model_forward=model_forward, samp_func=samp_func, style_forward=style_forward, latents_model=latents_model, dataloader=valid_dataloader,
                                                    args=args, device=device, sv_path=render_path, sigma_scale=args.sigma_scale)
        print('Done, saving', rgb_map.shape, t_map.shape)
        return

    # Render train style
    if args.render_train_style:
        render_path = os.path.join(sv_path, 'render_train_' + str(global_step))
        # Enable style
        model.set_enable_style(True)
        if args.N_samples_fine > 0:
            model_fine.set_enable_style(True)
        render_dataset = train_dataset
        render_dataset.mode = 'train_style'
        if args.N_samples_fine > 0:
            render_train_style(samp_func=samp_func, model_forward=model_forward, style_forward=style_forward, latents_model=latents_model, dataset=render_dataset, args=args, device=device, sv_path=render_path, model_forward_fine=model_forward_fine, samp_func_fine=samp_func_fine, sigma_scale=args.sigma_scale)
        else:
            render_train_style(samp_func=samp_func, model_forward=model_forward, style_forward=style_forward, latents_model=latents_model, dataset=render_dataset, args=args, device=device, sv_path=render_path, sigma_scale=args.sigma_scale)
        return

    # Training Loop
    def Origin_train(global_step):
        # Elapse Measurement
        data_time, model_time, opt_time = 0, 0, 0
        fine_time = 0
        while True:
            for batch_idx, batch_data in enumerate(train_dataloader):

                # To Device as Tensor
                for key in batch_data:
                    batch_data[key] = jt.array(batch_data[key].numpy())

                # Get batch data
                start_t = time.time()
                rgb_gt, rays_o, rays_d = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d']

                pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=train_dataset.near, far=train_dataset.far, perturb=True)
                ray_num, pts_num = rays_o.shape[0], args.N_samples
                rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])

                # Forward and Composition
                forward_t = time.time()
                ret = model_forward(pts=pts, dirs=rays_d_forward)
                pts_rgb, pts_sigma = ret['rgb'], ret['sigma']
                rgb_exp, t_exp, weights = alpha_composition(pts_rgb, pts_sigma, ts, args.sigma_noise_std)

                # Calculate Loss
                loss_rgb = img2mse(rgb_gt, rgb_exp)
                # opt_t = time.time()
                loss = loss_rgb

                fine_t = time.time()
                if args.N_samples_fine > 0:
                    pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
                    # print(pts_fine)
                    pts_num = args.N_samples + args.N_samples_fine
                    rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
                    ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
                    pts_rgb_fine, pts_sigma_fine = ret['rgb'], ret['sigma']
                    rgb_exp_fine, t_exp_fine, _ = alpha_composition(pts_rgb_fine, pts_sigma_fine, ts_fine, args.sigma_noise_std)
                    loss_rgb_fine = img2mse(rgb_gt, rgb_exp_fine)
                    loss = loss + loss_rgb_fine

                # Backward and Optimize
                optimizer.step(loss)

                if global_step % args.i_print == 0:
                    psnr = mse2psnr(loss_rgb)
                    if args.N_samples_fine > 0:
                        psnr_fine = mse2psnr(loss_rgb_fine)
                        tqdm.write(
                            f"[ORIGIN TRAIN] Iter: {global_step} Loss: {loss.data[0]} PSNR: {psnr.data[0]} PSNR Fine: {psnr_fine.data[0]} RGB Loss: {loss_rgb.data[0]} RGB Fine Loss: {loss_rgb_fine.data[0]}"
                            f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")
                        # tqdm.write(
                        #     f"[ORIGIN TRAIN] Iter: {global_step} Loss: {loss_rgb.item()} PSNR: {psnr.item()} RGB Loss: {loss_rgb.item()}"
                        #     f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")
                    else:
                        # tqdm.write(f"[ORIGIN TRAIN] Iter: {global_step}")
                        tqdm.write(
                            f"[ORIGIN TRAIN] Iter: {global_step} Loss: {loss_rgb.item()} PSNR: {psnr.item()} RGB Loss: {loss_rgb.item()}"
                            f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")

                    data_time, model_time, opt_time = 0, 0, 0
                    fine_time = 0

                # Update Learning Rate
                decay_rate = 0.1
                decay_steps = args.lrate_decay
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                # Time Measuring
                end_t = time.time()
                data_time += (forward_t - start_t)
                model_time += (fine_t - forward_t)
                fine_time += 0
                opt_time += (end_t - fine_t)

                # Rest is logging
                if global_step % args.i_weights == 0 and global_step > 0 or global_step >= args.origin_step:
                    path = os.path.join(ckpts_path, '{:06d}.tar'.format(global_step))
                    if args.N_samples_fine > 0:
                        jt.save({
                            'global_step': global_step,
                            'model': model.state_dict(),
                            'model_fine': model_fine.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'style_optimizer': style_optimizer.state_dict()
                        }, path)
                    else:
                        jt.save({
                            'global_step': global_step,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'style_optimizer': style_optimizer.state_dict()
                        }, path)
                    print('Saved checkpoints at', path)

                    # Delete ckpts
                    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f]
                    if len(ckpts) > args.ckp_num:
                        os.remove(ckpts[0])

                global_step += 1
                if global_step > args.origin_step:
                    return global_step

    def Style_train(global_step, train_dataset):
        # Elapse Measurement
        data_time, model_time, opt_time = 0, 0, 0
        fine_time = 0

        """VGG Net"""
        decoder = VGGNet.decoder
        vgg = VGGNet.vgg

        decoder_data = torch.load(sv_path+'/decoder.pth')
        if 'decoder' in decoder_data.keys():
            decoder.load_state_dict(decoder_data['decoder'])
        else:
            decoder.load_state_dict(decoder_data)
        vgg.load_state_dict(torch.load(args.vgg_pth_path))
        vgg = torch_nn.Sequential(*list(vgg.children())[:31])
        style_net = VGGNet.Net(vgg, decoder)
        # style_net.eval()
        style_net.to(device)
        # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-7)

        """Dataset Mode for Style"""
        if not type(train_dataset) is StyleRaySampler_gen:
            train_dataset = StyleRaySampler_gen(data_path=args.datadir, gen_path=nerf_gen_data_path, style_path=args.styledir, factor=args.factor,
                                                mode='train', valid_factor=args.valid_factor, dataset_type=args.dataset_type,
                                                white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc, TT_far=args.TT_far,
                                                pixel_alignment=args.pixel_alignment, spherify=args.spherify, decode_path=sv_path+'/decoder.pth', store_rays=args.store_rays)
        else:
            train_dataset.collect_all_stylized_images()
        train_dataset.set_mode('train_style')
        train_dataloader = LightDataLoader(train_dataset, batch_size=args.batch_size_style, shuffle=True, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
        rounds_per_epoch = int(train_dataloader.data_num / train_dataloader.batch_size)
        print('DataLoader Creation Done !')

        """Model Mode for Style"""
        model.set_enable_style(True)
        if args.N_samples_fine > 0:
            model_fine.set_enable_style(True)
        model.eval()

        latents_model.set_optimizer()

        patch_size = 512  # for 512 * 512 (or = 1024 for 1024 * 1024)
        resample_layer = torch_nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=True)

        loss_c, loss_s = torch.tensor(0.), torch.tensor(0.)
        while True:
            for _ in range(rounds_per_epoch):
                batch_data = train_dataloader.get_batch()
                # # Patch Sampling
                # if global_step > args.decoder_step:
                #     style_id, fid, hid, wid = np.random.randint(0, train_dataset.style_num), \
                #                               np.random.randint(0, train_dataset.frame_num), \
                #                               np.random.randint(0, train_dataset.h), \
                #                               np.random.randint(0, train_dataset.w)
                #     batch_data = train_dataset.get_patch_train_style(style_id=style_id, fid=fid, hid=hid, wid=wid, patch_size=patch_size)
                #     content_images = torch.movedim(batch_data['rgb_origin'].to(device).float().reshape([1, patch_size, patch_size, 3]), -1, 1)
                #     style_images = torch.movedim(batch_data['style_image'].to(device).float(), -1, 1)
                #     loss_c, loss_s, stylized_content = style_net(content_images, style_images,
                #                                                  return_stylized_content=True)
                #     loss_c, loss_s = args.content_loss_lambda * loss_c, args.style_loss_lambda * loss_s
                #     stylized_content = resample_layer(stylized_content)
                #
                #     samp_idx = np.random.choice(np.arange(patch_size ** 2), [args.batch_size_style], replace=False)
                #     batch_data['rgb_gt'] = torch.clip(torch.movedim(stylized_content, 1, -1).reshape([-1, 3])[samp_idx].detach(), 0, 1)
                #     rgb_2d = torch.clip(torch.movedim(stylized_content, 1, -1).reshape([-1, 3])[samp_idx], 0, 1)
                #
                #     samp_keys = ['rays_o', 'rays_d', 'frame_id', 'style_id', 'rgb_origin']
                #     for key in samp_keys:
                #         batch_data[key] = batch_data[key][samp_idx]

                # To Device as Tensor
                # To Device as Tensor
                for key in batch_data:
                    batch_data[key] = jt.array(batch_data[key].numpy())

                # Get batch data
                start_t = time.time()
                rgb_gt, rays_o, rays_d, rgb_origin = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d'], batch_data['rgb_origin']
                style_id, frame_id = batch_data['style_id'].long(), batch_data['frame_id'].long()

                # Sample
                pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=train_dataset.near, far=train_dataset.far, perturb=True)
                ray_num, pts_num = rays_o.shape[0], args.N_samples
                rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])

                # Forward
                forward_t = time.time()
                ret = model_forward(pts=pts, dirs=rays_d_forward)
                pts_sigma, pts_embed = ret['sigma'], ret['pts']
                # Stylize
                style_latents = latents_model(style_ids=style_id, frame_ids=frame_id)
                style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
                ret_style = style_forward(x=pts_embed, latent=style_latents_forward)
                pts_rgb_style = ret_style['rgb']
                # Composition
                rgb_exp_style, _, weights = alpha_composition(pts_rgb_style, pts_sigma, ts, args.sigma_noise_std)
                # Pixel-wise Loss
                loss_rgb = args.rgb_loss_lambda * img2mse(rgb_exp_style, rgb_gt)
                # Latent LogP loss
                logp_loss_lambda = args.logp_loss_lambda * (args.logp_loss_decay ** int((global_step - args.origin_step) / 1000))
                loss_logp = logp_loss_lambda * latents_model.minus_logp(style_ids=style_id, frame_ids=frame_id)

                fine_t = time.time()
                if args.N_samples_fine > 0:
                    # Sample
                    pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
                    pts_num = args.N_samples + args.N_samples_fine
                    rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
                    # Forward
                    ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
                    pts_sigma_fine, pts_embed_fine = ret['sigma'], ret['pts']
                    # Stylize
                    style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
                    ret_style = style_forward(x=pts_embed_fine, latent=style_latents_forward)
                    pts_rgb_style_fine = ret_style['rgb']
                    # Composition
                    rgb_exp_style_fine, _, _ = alpha_composition(pts_rgb_style_fine, pts_sigma_fine, ts_fine, args.sigma_noise_std)
                    loss_rgb_fine = args.rgb_loss_lambda * img2mse(rgb_exp_style_fine, rgb_gt)
                    loss_rgb += loss_rgb_fine

                # Loss for stylized NeRF
                loss_mimic = loss_rgb
                loss = loss_mimic + loss_logp

                # # Loss for 2D stylization network
                # if global_step > args.decoder_step:
                #     loss_rgb_2d = args.rgb_loss_lambda_2d * img2mse(rgb_2d, rgb_exp_style.detach())
                #     if args.N_samples_fine > 0:
                #         loss_rgb_2d_fine = args.rgb_loss_lambda_2d * img2mse(rgb_2d, rgb_exp_style_fine.detach())
                #         loss_rgb_2d += loss_rgb_2d
                #     loss_mimic = loss_rgb_2d
                #     loss_2d = loss_mimic + loss_c + loss_s

                # Backward and Optimize
                opt_t = time.time()
                # if global_step > args.decoder_step:
                #     loss_2d.backward()
                #     decoder_optimizer.step()
                style_optimizer.step(loss)
                latents_model.optimize(loss)

                # Update Learning Rate
                decay_rate = 0.1
                decay_steps = args.lrate_decay
                new_lrate = args.lrate * (decay_rate ** ((global_step - args.origin_step) / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                # Time Measuring
                end_t = time.time()
                data_time += (forward_t - start_t)
                model_time += (fine_t - forward_t)
                fine_time += (opt_t - fine_t)
                opt_time += (end_t - fine_t)

                # Rest is logging
                if global_step % args.i_weights == 0 and global_step > 0:
                    # Saving Style module
                    path = os.path.join(ckpts_path, 'style_{:06d}.tar'.format(global_step))
                    jt.save({
                        'global_step': global_step,
                        'model': style_model.state_dict(),
                        'optimizer': style_optimizer.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)
                    # Delete ckpts
                    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' in f and 'latent' not in f]
                    if len(ckpts) > args.ckp_num:
                        os.remove(ckpts[0])

                    # Saving Latent Model
                    path = os.path.join(ckpts_path, 'latent_{:06d}.tar'.format(global_step))
                    jt.save({
                        'global_step': global_step,
                        'train_set': latents_model.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)
                    # Delete ckpts
                    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' not in f and 'latent' in f]
                    if len(ckpts) > args.ckp_num:
                        os.remove(ckpts[0])

                    # # Saving 2D stylization method
                    # state_dict = style_net.decoder.state_dict()
                    # for key in state_dict.keys():
                    #     state_dict[key] = state_dict[key]
                    # sv_dict = {'decoder': state_dict, 'step': global_step}
                    # torch.save(sv_dict, ckpts_path + '/decoder.pth')

                if global_step % args.i_print == 0:
                    tqdm.write(f"[STYLE TRAIN] Iter: {global_step} Loss: {loss.item()} Pixel RGB Loss: {loss_rgb.item()} -Log(p) Loss: {loss_logp.item()} Loss Content: {loss_c.item()} Loss Style: {loss_s.item()}"
                               f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")
                    data_time, model_time, opt_time = 0, 0, 0
                    fine_time = 0

                global_step += 1
                if global_step > args.total_step:
                    return global_step

    if global_step + 1 < args.origin_step:
        print('Global Step: ', global_step, ' Origin Step: ', args.origin_step)
        print('Origin Train')
        Origin_train(global_step)
    else:
        sv_name = '/decoder.pth'
        is_ndc = (args.dataset_type == 'llff' and not args.no_ndc)
        if not os.path.exists(sv_path + sv_name):
            if args.dataset_type == 'llff':
                train_temporal_invoke(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/', is_ndc=is_ndc,
                                      nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)
            else:
                train_temporal_invoke_pl(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
                                         nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)

        if not os.path.exists(nerf_gen_data_path):
            Prepare_Style_data(nerf_gen_data_path=nerf_gen_data_path)

        Style_train(global_step, train_dataset)
        exit(0)
    return


if __name__ == '__main__':
    args = config_parser()
    while True:
        train(args=args)
