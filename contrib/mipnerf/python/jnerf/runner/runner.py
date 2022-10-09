import os
import jittor as jt
from PIL import Image
import numpy as np
from jnerf.dataset import namedtuple_map
from tqdm import tqdm
from jnerf.utils.config import get_cfg, save_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, SCHEDULERS, DATASETS, OPTIMS, SAMPLERS, LOSSES
from jnerf.models.losses.mse_loss import img2mse, mse2psnr
from jnerf.utils.logs import get_log


class MipRunner:
    def __init__(self):
        self.cfg = get_cfg()
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        self.exp_name = self.cfg.exp_name
        self.dataset = {}
        self.dataset["train"] = build_from_cfg(self.cfg.dataset.train, DATASETS)
        self.cfg.dataset_obj = self.dataset["train"]
        self.dataset["val"] = build_from_cfg(self.cfg.dataset.val, DATASETS)
        self.dataset["test"] = build_from_cfg(self.cfg.dataset.test, DATASETS)
        self.model = build_from_cfg(self.cfg.model, NETWORKS)
        self.cfg.model_obj = self.model
        self.sampler = build_from_cfg(self.cfg.sampler, SAMPLERS)
        self.cfg.sampler_obj = self.sampler
        self.optimizer = build_from_cfg(self.cfg.optim, OPTIMS, params=self.model.parameters())
        self.optimizer = build_from_cfg(self.cfg.linearlog, OPTIMS, nested_optimizer=self.optimizer)
        # self.ema_optimizer      = build_from_cfg(self.cfg.ema, OPTIMS, params=self.model.parameters())
        self.loss_func = build_from_cfg(self.cfg.loss, LOSSES)
        #
        self.background_color = self.cfg.background_color
        self.tot_train_steps = self.cfg.tot_train_steps
        self.cfg.m_training_step = 0
        self.n_rays_per_batch = self.cfg.n_rays_per_batch
        self.using_fp16 = self.cfg.using_fp16
        self.save_path = os.path.join(self.cfg.log_dir, self.exp_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.cfg.ckpt_path and self.cfg.ckpt_path is not None:
            self.ckpt_path = self.cfg.ckpt_path
        else:
            self.ckpt_path = os.path.join(self.save_path, "params.pkl")
        if self.cfg.load_ckpt:
            self.load_ckpt(self.ckpt_path)
        else:
            self.start = 0
        # some mip-nerf para
        self.num_levels = self.cfg.num_levels  # 采样几层
        self.coarse_loss_mult = self.cfg.coarse_loss_mult  # 权重
        self.disable_multiscale_loss = self.cfg.disable_multiscale_loss

        # 测试集
        self.chunk = 3072

    def get_rgb_density(self, rays):
        ret = []
        t_vals, weights = None, None
        for i_level in range(self.num_levels):
            samples_enc, viewdirs_enc, t_vals = self.sampler.sample(rays, i_level, t_vals, weights)
            raw_rgb, raw_density = self.model(samples_enc, viewdirs_enc)  # 放入MLP
            comp_rgb, distance, acc, weights = self.sampler.rays2rgb(rays, raw_rgb, raw_density, t_vals)  # 体素渲染
            ret.append((comp_rgb, distance, acc))
        return ret

    def train(self):
        # 打印loss
        try:
            os.mknod(os.path.join(self.save_path, 'loss.txt'))
        except:
            pass
        logger = get_log(os.path.join(self.save_path, 'loss.txt'))
        print("开始训练")
        for i in tqdm(range(self.start, self.tot_train_steps)):
            '''
            rays：所有光线的七个属性
            rgb_target: 取到的像素, 范围归一化[0, 1] 默认四通道
            '''
            self.cfg.m_training_step = i
            rays, rgb_target = next(self.dataset["train"])
            ret = self.get_rgb_density(rays)  # (comp_rgb [bs ,3], distance, acc),  coarse网络对应list[0] fine是1
            mask = rays.lossmult
            if self.disable_multiscale_loss:
                mask = jt.ones_like(mask)
            # all level's results will contribute to final loss
            losses = []
            for (rgb, _, _) in ret:
                losses.append(
                    (mask * (rgb - rgb_target[..., :3]) ** 2).sum() / mask.sum())
                # losses.append(self.loss_func(rgb, rgb_target) / mask.sum())
            loss = self.coarse_loss_mult * jt.sum(losses[:-1]) + losses[-1]  # coarse网络 + fine网络
            if i % 100 == 0 and i > 0:
                logger.info('train step {} loss：{}'.format(i, loss.item()))
            self.optimizer.step(loss)
            if self.using_fp16:
                self.model.set_fp16()
            # if i == 5 and i > 0:
            #     psnr = mse2psnr(self.val_img(i))
            #     logger.info("STEP={} | LOSS={} | VAL PSNR={}".format(i, loss.mean().item(), psnr))
            if i % 2000 == 0 and i > 0:
                psnr = mse2psnr(self.val_img(i))
                logger.info("STEP={} | LOSS={} | VAL PSNR={}".format(i, loss.mean().item(), psnr))
            if i % 5000 == 0 and i > 0:
                self.test(False)
        self.save_ckpt(os.path.join(self.save_path, "params.pkl"))

    def test(self, load_ckpt=False):
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: " + self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if self.dataset["test"] is None:
            self.dataset["test"] = build_from_cfg(self.cfg.dataset.test, DATASETS)
        if not os.path.exists(os.path.join(self.save_path, "test")):
            os.makedirs(os.path.join(self.save_path, "test"))
        mse_list = self.render_test(save_path=os.path.join(self.save_path, "test"))
        if self.dataset["test"].have_img:
            tot_psnr = 0
            for mse in mse_list:
                tot_psnr += mse2psnr(mse)
            print("TOTAL TEST PSNR===={}".format(tot_psnr / len(mse_list)))

    def save_ckpt(self, path):
        jt.save({
            'global_step': self.cfg.m_training_step,
            'model': self.model.state_dict(),
            'sampler': self.sampler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'nested_optimizer': self.optimizer.nested_optimizer.state_dict(),
            # 'ema_optimizer': self.ema_optimizer.state_dict(),
        }, path)

    def load_ckpt(self, path):
        print("Loading ckpt from:", path)
        ckpt = jt.load(path)
        self.start = ckpt['global_step']
        self.model.load_state_dict(ckpt['model'])
        if self.using_fp16:
            self.model.set_fp16()
        self.sampler.load_state_dict(ckpt['sampler'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        nested = ckpt['nested_optimizer']['defaults']['param_groups'][0]
        for pg in self.optimizer._nested_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i] = jt.array(nested["values"][i])
                pg["m"][i] = jt.array(nested["m"][i])
        ema = ckpt['ema_optimizer']['defaults']['param_groups'][0]
        for pg in self.ema_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i] = jt.array(ema["values"][i])
        self.ema_optimizer.steps = ckpt['ema_optimizer']['defaults']['steps']

    def val_img(self, iter):
        with jt.no_grad():
            img, img_tar = self.render_img(dataset_mode="val")
            self.save_img(self.save_path + f"/img{iter}.png", img)
            self.save_img(self.save_path + f"/target{iter}.png", img_tar)
            return img2mse(
                jt.array(img),
                jt.array(img_tar)).item()

    def render_test(self, save_img=True, save_path=None):
        if save_path is None:
            save_path = self.save_path
        mse_list = []
        for img_i in tqdm(range(0, self.dataset["test"].n_examples, 1)):
            with jt.no_grad():
                imgs = []
                for i in range(1):
                    simg, img_tar = self.render_img(dataset_mode="test", img_id=img_i)  # 这里用了许多的render_img
                    imgs.append(simg)
                img = np.stack(imgs, axis=0).mean(0)
                if save_img:
                    self.save_img(save_path + f"/{self.exp_name}_r_{img_i}.png", img)
                    self.save_img(save_path + f"/{self.exp_name}_gt_{img_i}.png", img_tar)
                mse_list.append(img2mse(
                    jt.array(img),
                    jt.array(img_tar)).item())
        return mse_list

    def save_img(self, path, img):
        if isinstance(img, np.ndarray):
            ndarr = (img * 255 + 0.5).clip(0, 255).astype('uint8')
        elif isinstance(img, jt.Var):
            ndarr = (img * 255 + 0.5).clamp(0, 255).uint8().numpy()
        im = Image.fromarray(ndarr)
        im.save(path)

    def flattens(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        x = np.concatenate(x, axis=0)
        return jt.array(x)

    def render_img(self, dataset_mode=None, img_idx=None):
        print("begin {}_model".format(dataset_mode))
        if img_idx is None:
            print("extract only one image from val_datasets")
            img_idx = np.random.randint(0, self.dataset[dataset_mode].n_examples, [1])[0]
        else:
            img_idx = img_idx
        rgb_target = self.dataset[dataset_mode].images[img_idx]
        rays = namedtuple_map(lambda r: jt.array(r[img_idx]),
                              self.dataset[dataset_mode].rays)  # rays是tup   # 不调用next，每次取1个图及光线
        rays = namedtuple_map(self.flattens, rays)
        rgb_target = rgb_target[..., :3]  # 保留3通道
        height, width, _ = rgb_target.shape
        num_rays = int(height * width)
        results = []
        for i in range(0, num_rays, self.chunk):
            # pylint: disable=cell-var-from-loop
            chunk_rays = namedtuple_map(lambda r: r[i:i + self.chunk], rays)
            rays_per_host = chunk_rays.origins.shape[0]
            start, stop = 0, 1 * rays_per_host
            chunk_rays = namedtuple_map(lambda r: r[start:stop], chunk_rays)
            # 取fine网络 (comp_rgb [bs ,3], distance, acc),  coarse网络对应list[0] fine是1
            chunk_results = self.get_rgb_density(chunk_rays)[-1]
            results.append(chunk_results)
        rgb, distance, acc = [jt.concat(r, dim=0) for r in zip(*results)]  # reshape成图片的形状
        rgb = rgb.reshape((height, width, -1))
        distance = distance.reshape((height, width))
        acc = acc.reshape((height, width))
        jt.sync_all()
        jt.gc()
        return rgb, rgb_target
