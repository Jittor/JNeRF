import json
import os
from symbol import factor

from jnerf.models.networks.svox2_network import SparseGrid
import jittor as jt
from PIL import Image
import numpy as np
from tqdm import tqdm
from jnerf.ops.code_ops import *
from jnerf.dataset.dataset import jt_srgb_to_linear, jt_linear_to_srgb
from jnerf.utils.config import get_cfg, save_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES
from jnerf.models.losses.mse_loss import img2mse, mse2psnr
from jnerf.dataset import camera_path
from jnerf.optims.svox2_optim import PlenOptimRMSprop
import cv2
import math
from jnerf.utils.svox2_utils import *

class Svox2Runner():
    def __init__(self):
        jt.set_global_seed(20200823)
        self.cfg = get_cfg()
   
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        self.exp_name           = self.cfg.exp_name
        self.dataset            = {}
        self.dataset["train"]   = build_from_cfg(self.cfg.dataset.train, DATASETS)
        self.cfg.dataset_obj    = self.dataset["train"]
        if self.cfg.dataset.val:
            self.dataset["val"] = build_from_cfg(self.cfg.dataset.val, DATASETS)
        else:
            self.dataset["val"] = self.dataset["train"]
        self.dataset["test"]    = None
        self.reso_list = self.cfg.reso_list
        reso_id = 0

        self.model = SparseGrid(self.reso_list[reso_id],self.dataset['train'].scene_radius,
        self.dataset['train'].scene_center,1,self.cfg.model.basis_dim,self.cfg.model.basis_reso,
        use_z_order=True,use_sphere_bound=self.dataset['train'].use_sphere_bound and not self.cfg.nosphereinit)
        self.cfg.model_obj      = self.model
        self.optimizer = optimizer = PlenOptimRMSprop(self.model.density_data,self.model.sh_data,0,0,0.95,0.95)

        self.loss_func          = build_from_cfg(self.cfg.loss, LOSSES)
        self.background_color   = self.cfg.background_color
        self.tot_train_steps    = self.cfg.tot_train_steps
        self.n_rays_per_batch   = self.cfg.n_rays_per_batch
        self.using_fp16         = self.cfg.fp16
        self.save_path          = os.path.join(self.cfg.log_dir, self.exp_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.cfg.ckpt_path and self.cfg.ckpt_path is not None:
            self.ckpt_path = self.cfg.ckpt_path
        else:
            self.ckpt_path = os.path.join(self.save_path, "ckpt.npz")
        if self.cfg.load_ckpt:
            self.load_ckpt(self.ckpt_path)
        else:
            self.start=0
        self.alpha_image=self.cfg.alpha_image

        self.cfg.m_training_step = 0
        self.val_freq = 4096
        self.factor = 1




    def train(self):
        args = self.cfg
        factor = self.factor
        self.model.param_init(args)
        lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
        lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                                    args.lr_sh_delay_mult, args.lr_sh_decay_steps)
     
        lr_sigma_factor = 1.0
        lr_sh_factor = 1.0

        last_upsamp_step = args.init_iters
        epoch_id = -1
        dset=self.dataset["train"]
        dset_test=self.dataset['val']
        grid=self.model
        reso_list = self.reso_list
        reso_id = 0
        gstep_id_base = 0

        resample_cameras = [
                Camera(c2w,
                            dset.intrins.get('fx', i),
                            dset.intrins.get('fy', i),
                            dset.intrins.get('cx', i),
                            dset.intrins.get('cy', i),
                            width=dset.get_image_size(i)[1],
                            height=dset.get_image_size(i)[0],
                            ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
            ]
        while True:
                
            dset.shuffle_rays()
            epoch_id += 1
            epoch_size = dset.rays.origins.size(0)
            batches_per_epoch = (epoch_size-1)//args.batch_size+1

            # Test
            def eval_step():
                # Put in a function to avoid memory leak
                print('Eval step')
                with jt.no_grad():
                    stats_test = {'psnr' : 0.0, 'mse' : 0.0}

                    # Standard set
                    N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
                    N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
                    img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
                    img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
                    img_ids = range(0, dset_test.n_images, img_eval_interval)


                    n_images_gen = 0
                    for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                        c2w = dset_test.c2w[img_id]
                        cam = Camera(c2w,
                                        dset_test.intrins.get('fx', img_id),
                                        dset_test.intrins.get('fy', img_id),
                                        dset_test.intrins.get('cx', img_id),
                                        dset_test.intrins.get('cy', img_id),
                                        width=dset_test.get_image_size(img_id)[1],
                                        height=dset_test.get_image_size(img_id)[0],
                                        ndc_coeffs=dset_test.ndc_coeffs)
                        rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                        rgb_gt_test = dset_test.gt[img_id]
                        all_mses = ((rgb_gt_test - rgb_pred_test) ** 2)
                        if i % img_save_interval == 0:
                            img_pred = rgb_pred_test
                            img_pred=jt.clamp(img_pred,max_v=1.0)
                            img_dir = self.save_path
                            img_path = os.path.join(img_dir,f"{gstep_id_base:09d}")
                            if not os.path.exists(img_path):
                                os.makedirs(img_path)
                            jt.save_image(img_pred.permute(2,0,1),img_path+f"/image_pred_{img_id:04d}.png")
                            jt.save_image(rgb_gt_test.permute(2,0,1),img_path+f"/image_test_{img_id:04d}.png")
                        rgb_pred_test = rgb_gt_test = None
                        mse_num : float = all_mses.mean().item()
                        psnr = -10.0 * math.log10(mse_num)
                        if math.isnan(psnr):
                            print('NAN PSNR', i, img_id, mse_num)
                            assert False
                        stats_test['mse'] += mse_num
                        stats_test['psnr'] += psnr
                        n_images_gen += 1

                    

                    stats_test['mse'] /= n_images_gen
                    stats_test['psnr'] /= n_images_gen
                    print('eval stats:', stats_test)
            if epoch_id % max(factor, args.eval_every) == 0: #and (epoch_id > 0 or not args.tune_mode):
                # NOTE: we do an eval sanity check, if not in tune_mode
                eval_step()
                # print("eval step")
                jt.gc()

            def train_step():
                print('Train step')
                pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
                stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
                for iter_id, batch_begin in pbar:
                    gstep_id = iter_id + gstep_id_base
                    if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                        grid.density_data.data[:] = args.init_sigma
                    lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
                    lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
                    if not args.lr_decay:
                        lr_sigma = args.lr_sigma * lr_sigma_factor
                        lr_sh = args.lr_sh * lr_sh_factor


                    batch_end = min(batch_begin + args.batch_size, epoch_size)
                    batch_origins = dset.rays.origins[batch_begin: batch_end]
                    batch_dirs = dset.rays.dirs[batch_begin: batch_end]
                    rgb_gt = dset.rays.gt[batch_begin: batch_end]
                    rays = Rays(batch_origins, batch_dirs)
    
                    rgb_pred=grid.volume_render_jt(rays,randomize=args.enable_random)


                    mse = self.loss_func(rgb_gt, rgb_pred)
                  
                    mse_num : float = mse.detach().item()
                    psnr = -10.0 * math.log10(mse_num)
                    stats['mse'] += mse_num
                    stats['psnr'] += psnr
                    stats['invsqr_mse'] += 1.0 / mse_num ** 2
                    self.optimizer.zero_grad()
                    self.optimizer.backward(mse)
            
                    self.optimizer.update_lr(lr_sigma,lr_sh,args.rms_beta,args.rms_beta)
                    
                    grad_density_data=grid.density_data.opt_grad(self.optimizer)
                 
                    grad_sh_data=grid.sh_data.opt_grad(self.optimizer)
                    
                    
                    if (iter_id + 1) % args.print_every == 0:
                        # Print averaged stats
                        pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')

                  

                        if args.weight_decay_sh < 1.0:
                            grid.sh_data.data *= args.weight_decay_sigma
                        if args.weight_decay_sigma < 1.0:
                            grid.density_data.data *= args.weight_decay_sh

                  

                    # Apply TV/Sparsity regularizers
            
                    if args.lambda_tv > 0.0:

                        grad_density_data.assign(grid.inplace_tv_grad(grad_density_data,
                                scaling=args.lambda_tv,
                                sparse_frac=args.tv_sparsity,
                                logalpha=args.tv_logalpha,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=args.tv_contiguous))
                    if args.lambda_tv_sh > 0.0:
                      
                        grad_sh_data.assign(grid.inplace_tv_color_grad(grad_sh_data,
                                scaling=args.lambda_tv_sh,
                                sparse_frac=args.tv_sh_sparsity,
                                ndc_coeffs=dset.ndc_coeffs,
                                contiguous=args.tv_contiguous))
                
                    self.optimizer.step()
   

                  
            train_step()
            jt.gc()
            gstep_id_base += batches_per_epoch
            if (gstep_id_base - last_upsamp_step) >= args.upsamp_every:

                last_upsamp_step = gstep_id_base
                if reso_id < len(reso_list) - 1:
                    print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
                    if args.tv_early_only > 0:
                        print('turning off TV regularization')
                        args.lambda_tv = 0.0
                        args.lambda_tv_sh = 0.0
                    elif args.tv_decay != 1.0:
                        args.lambda_tv *= args.tv_decay
                        args.lambda_tv_sh *= args.tv_decay

                    reso_id += 1
                    use_sparsify = True
                    z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
                    grid.resample(reso=reso_list[reso_id],
                            sigma_thresh=args.density_thresh,
                            weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                            dilate=2, #use_sparsify,
                            cameras=resample_cameras if args.thresh_type == 'weight' else None,
                            max_elements=args.max_grid_elements)
                    del self.optimizer
                    self.optimizer = PlenOptimRMSprop(grid.density_data,grid.sh_data,0,0,0.95,0.95)


                    if args.upsample_density_add:
                        grid.density_data.data[:] += args.upsample_density_add

                if factor > 1 and reso_id < len(reso_list) - 1:
                    print('* Using higher resolution images due to large grid; new factor', factor)
                    factor //= 2
                    dset.gen_rays(factor=factor)
                    dset.shuffle_rays()
            if gstep_id_base >= args.n_iters:
                print('* Final eval and save')
                eval_step()
                grid.save(self.ckpt_path)
                break


    def test(self, load_ckpt=False):  
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if self.dataset["test"] is None:
            self.dataset["test"] = build_from_cfg(self.cfg.dataset.test, DATASETS)
            self.dataset["test"].have_img = True

        if not os.path.exists(os.path.join(self.save_path, "test")):
            os.makedirs(os.path.join(self.save_path, "test"))
        mse_list=self.render_test(save_path=os.path.join(self.save_path, "test"))
        if self.dataset["test"].have_img:
            tot_psnr=0
            for mse in mse_list:
                tot_psnr += mse2psnr(mse)
            print("TOTAL TEST PSNR===={}".format(tot_psnr/len(mse_list)))

       
    def save_ckpt(self, path):
        self.model.save(path)


    def load_ckpt(self, path):
        print("Loading ckpt from:",path)
        del self.model
        self.model = SparseGrid.load(path)
        jt.sync_all(True)


    def render_test(self, save_img=True, save_path=None):
        if save_path is None:
            save_path = self.save_path
        mse_list = []
        print("rendering testset...")
        for img_i in tqdm(range(0,self.dataset["test"].n_images,1)):
            with jt.no_grad():
                imgs=[]
                for i in range(1):
                    simg,img_tar = self.render_img(dataset_mode="test", img_id=img_i)
                    imgs.append(simg)
                img = np.stack(imgs, axis=0).mean(0)
                if save_img:
                    self.save_img(save_path+f"/{self.exp_name}_r_{img_i}.png", img)
                    if self.dataset["test"].have_img:
                        self.save_img(save_path+f"/{self.exp_name}_gt_{img_i}.png", img_tar)
                mse_list.append(img2mse(
                jt.array(img), 
                jt.array(img_tar)).item())
        return mse_list


    def save_img(self, path, img, alpha=None):
        if alpha is not None:
            img = np.concatenate([img, alpha], axis=-1)
        if isinstance(img, np.ndarray):
            ndarr = (img*255+0.5).clip(0, 255).astype('uint8')
        elif isinstance(img, jt.Var):
            ndarr = (img*255+0.5).clamp(0, 255).uint8().numpy()
        im = Image.fromarray(ndarr)
        im.save(path)


    def render_img(self, dataset_mode="train", img_id=None):
        dset_test = self.dataset['test']
        c2w = dset_test.c2w[img_id]
        cam = Camera(c2w,
                        dset_test.intrins.get('fx', img_id),
                        dset_test.intrins.get('fy', img_id),
                        dset_test.intrins.get('cx', img_id),
                        dset_test.intrins.get('cy', img_id),
                        width=dset_test.get_image_size(img_id)[1],
                        height=dset_test.get_image_size(img_id)[0],
                        ndc_coeffs=dset_test.ndc_coeffs)
        rgb_pred_test = self.model.volume_render_image(cam)
        rgb_gt_test = dset_test.gt[img_id]
        return rgb_pred_test, rgb_gt_test

