import os
import torch
import shutil
import VGGNet
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from models_jt import VAE
from pathlib import Path
from models_jt import Camera
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from plyfile import PlyElement, PlyData
from Style_function import calc_mean_std

# from pytorch3d.structures import Pointclouds
# from pytorch3d.renderer import compositing
# from pytorch3d.renderer.points import rasterize_points

# cudnn.benchmark = True
# Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# # Disable OSError: image file is truncated
# ImageFile.LOAD_TRUNCATED_IMAGES = True


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        # super(InfiniteSamplerWrapper, self).__init__()
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def train_transform2():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def default_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform=None):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        transform = default_transform() if transform is None else transform
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


class CoorImageDataset(data.Dataset):
    def __init__(self, root):
        super(CoorImageDataset, self).__init__()
        self.root = root
        self.image_paths = sorted(list(Path(self.root).glob('rgb_*.png')))
        self.geo_paths = sorted(list(Path(self.root).glob('geometry_*.npz')))
        data = np.load(str(self.geo_paths[0]))
        self.hwf = data['hwf']
        # self.near, self.far = data['near'], data['far']
        self.near, self.far = 0., 1.
        self.transform = default_transform()

    def __getitem__(self, index):
        image_path, geo_path = self.image_paths[index], self.geo_paths[index]
        img = Image.open(str(image_path)).convert('RGB')
        img = self.transform(img)
        geo = np.load(str(geo_path))
        coor_map, cps = geo['coor_map'], geo['cps']
        return img, coor_map, cps

    def __len__(self):
        return len(self.image_paths)

    def name(self):
        return 'FlatFolderDataset'


class CoorImageDataset_pl(data.Dataset):
    def __init__(self, root, factor=0.01):
        super(CoorImageDataset_pl, self).__init__()
        self.root = root
        self.image_paths = sorted(list(Path(self.root).glob('rgb_*.png')))
        self.geo_paths = sorted(list(Path(self.root).glob('geometry_*.npz')))
        data = np.load(str(self.geo_paths[0]))
        self.hwf = data['hwf']
        # self.near, self.far = data['near'], data['far']
        self.near, self.far = 0., 1.
        self.factor = factor
        self.transform = default_transform()

        ts = np.zeros([len(self.geo_paths), 3], dtype=np.float32)
        for i in range(len(self.geo_paths)):
            ts[i] = np.load(str(self.geo_paths[i]))['cps'][:3, 3]

        dist = ts[np.newaxis] - ts[:, np.newaxis]
        dist = dist ** 2
        dist = dist.sum(-1) ** 0.5
        self.dist = dist

    def get_batch(self, batch_size, index=None):
        if index is None:
            index = np.random.randint(0, len(self.image_paths))
        dists = self.dist[index]
        inds = np.argsort(dists)
        prange = max(int(self.factor*len(self.image_paths)), batch_size)
        inds = inds[:prange]
        inds = np.random.choice(inds, [batch_size], replace=(prange <= batch_size))
        imgs, coor_maps, cps = [], [], []
        for i in range(batch_size):
            img, coor_map, cp = self.__getitem__(inds[i])
            imgs.append(img)
            coor_maps.append(coor_map)
            cps.append(cp)
        imgs = torch.stack(imgs).float()
        coor_maps = torch.from_numpy(np.stack(coor_maps)).float()
        cps = torch.from_numpy(np.stack(cps)).float()
        return imgs, coor_maps, cps

    def __getitem__(self, index):
        image_path, geo_path = self.image_paths[index], self.geo_paths[index]
        img = Image.open(str(image_path)).convert('RGB')
        img = self.transform(img)
        geo = np.load(str(geo_path))
        coor_map, cps = geo['coor_map'], geo['cps']
        return img, coor_map, cps

    def __len__(self):
        return len(self.image_paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def finetune_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    decoder = VGGNet.decoder
    vgg = VGGNet.vgg

    decoder.load_state_dict(torch.load('./models/decoder.pth'))
    vgg.load_state_dict(torch.load('./models/vgg_normalised.pth'))

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = VGGNet.Net(vgg, decoder)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        loss_c, loss_s = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()


def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    vgg = VGGNet.vgg
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.eval()
    vgg.to(device)

    style_tf = train_transform()
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    vae = VAE(data_dim=1024, latent_dim=args.vae_latent, W=args.vae_w, D=args.vae_d, kl_lambda=args.vae_kl_lambda)
    vae.train()
    vae.to(device)
    vae_ckpt = './pretrained/vae.pth'
    if os.path.exists(vae_ckpt):
        vae.load_state_dict(torch.load(vae_ckpt))
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i)
        style_images = next(style_iter).to(device)
        style_features = vgg(style_images)
        style_mean, style_std = calc_mean_std(style_features)
        style_features = torch.cat([style_mean.squeeze(), style_std.squeeze()], dim=-1)
        recon, _, mu, logvar = vae(style_features)
        loss, recon_loss, kl_loss = vae.loss(style_features, recon, mu, logvar, return_losses=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Reconstruction Loss', recon_loss.item(), i + 1)
        writer.add_scalar('KL Loss', kl_loss.item(), i + 1)

        if (i + 1) % 100 == 0:
            print("Loss: %.3f | Recon Loss: %.3f| KL Loss: %.3f" % (loss.item(), recon_loss.item(), kl_loss.item()))

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = vae.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, vae_ckpt)
    writer.close()


def train_temporal_invoke(save_dir, sv_name, log_dir, is_ndc, nerf_content_dir, style_dir, batch_size, n_threads=8, lr=1e-3, max_iter=1000):
    if is_ndc:
        print("Using NDC Coordinate System! Check Nerf and dataset to be LLFF !!!!!!!")
        temporal_weight, content_weight, style_weight = 50., 1.0, 1.
    else:
        temporal_weight, content_weight, style_weight = 50., 1.0, 1.

    print_interval = 20
    save_model_interval = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    save_dir, log_dir = str(save_dir), str(log_dir)

    decoder = VGGNet.decoder
    vgg = VGGNet.vgg

    ckpts = [os.path.join(save_dir, f) for f in sorted(os.listdir(save_dir)) if sv_name in f]
    if len(ckpts) > 0:
        ld_dict = torch.load(ckpts[-1])
        decoder.load_state_dict(ld_dict['decoder'])
        step = ld_dict['step']
    else:
        print('From original pth file')
        decoder.load_state_dict(torch.load('./pretrained/decoder.pth'))
        shutil.copy('./pretrained/decoder.pth', save_dir + '/' + sv_name)
        step = 0
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = VGGNet.Net(vgg, decoder)
    network.train()
    network.to(device)

    style_tf = train_transform2()

    content_dataset = CoorImageDataset(nerf_content_dir)
    style_dataset = FlatFolderDataset(style_dir, style_tf)

    # Camera for Rendering
    h, w, focal = content_dataset.hwf
    h, w = int(h), int(w)
    cx, cy = w/2, h/2
    near_prj, far_prj = 1e-3, 1e5
    projectionMatrix = np.array([[-2*focal/w, 0,          1-2*cx/w,               0],
                                 [0,          2*focal/h,  2*cy/h-1,               0],
                                 [0,          0,          -(far_prj+near_prj)/(far_prj-near_prj), -2*far_prj*near_prj/(far_prj-near_prj)],
                                 [0,          0,          -1,                     0]])
    camera = Camera(projectionMatrix=projectionMatrix)
    camera.to(device)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=1,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=n_threads))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=lr)

    space_dist_threshold = 5e-2

    def adjust_learning_rate_local(optimizer, iteration_count):
        """Imitating the original implementation"""
        lr = 1e-4 / (1.0 + 5e-5 * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for i in tqdm(range(step, max_iter)):
        # Sampling Patch
        patch_size = 512
        if patch_size > 0:
            patch_h_min, patch_w_min = np.random.randint(0, h - patch_size), np.random.randint(0, w - patch_size)
            patch_h_max, patch_w_max = patch_h_min + patch_size, patch_w_min + patch_size
        else:
            patch_h_min, patch_w_min = 0, 0
            patch_h_max, patch_w_max = h, w

        resample_layer = nn.Upsample(size=(int(patch_h_max - patch_h_min), int(patch_w_max - patch_w_min)),
                                     mode='bilinear', align_corners=True)

        adjust_learning_rate_local(optimizer, iteration_count=i)
        content_images, coor_maps, cps = next(content_iter)
        content_images, coor_maps, cps = content_images[..., patch_h_min: patch_h_max, patch_w_min: patch_w_max].to(device),\
                                         coor_maps[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max].to(device),\
                                         cps.to(device)
        if is_ndc:
            coor_maps = ndc2world(coor_maps, h, w, focal)

        # The same style image
        style_images = next(style_iter).to(device)
        style_images = style_images[:1].expand([batch_size, * style_images.shape[1:]])

        loss_c, loss_s, stylized_content = network(content_images, style_images, return_stylized_content=True)
        stylized_content = resample_layer(stylized_content)

        # Set camera pose
        camera.set(cameraPose=cps)
        pcl_coor_world0 = coor_maps[0].reshape([-1, 3])
        pcl_rgb0 = torch.movedim(stylized_content[0], 0, -1).reshape([-1, 3])

        warped_stylized_content0, warped_coor_map0, warped_msks = camera.rasterize(pcl_coor_world0, pcl_rgb0, h=h, w=w)
        warped_stylized_content0, warped_coor_map0, warped_msks = warped_stylized_content0[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max],\
                                                                  warped_coor_map0[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max],\
                                                                  warped_msks[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max]
        coor_dist_msk = (((warped_coor_map0 - coor_maps) ** 2).sum(-1, keepdim=True) < space_dist_threshold ** 2).float()

        loss_t = (((torch.movedim(stylized_content, 1, -1) - warped_stylized_content0) ** 2) * warped_msks * coor_dist_msk).mean()
        loss_t = temporal_weight * loss_t

        loss_c = content_weight * loss_c
        loss_s = style_weight * loss_s
        loss = loss_c + loss_s + loss_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_temporal', loss_t.item(), i + 1)

        if (i + 1) % print_interval == 0:
            print('Iter %d Content Loss: %.3f Style Loss: %.3f Temporal Loss: %.3f' % (i, loss_c.item(), loss_s.item(), loss_t.item()))

        if i == 0 or (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            sv_dict = {'decoder': state_dict, 'step': (i+1)}
            torch.save(sv_dict, save_dir + '/' + sv_name)

            warped_stylized_content0 = torch.clamp(warped_stylized_content0, 0, 1).detach().cpu().numpy()
            coor_dist_msk = np.broadcast_to(coor_dist_msk.detach().cpu().numpy(), [*coor_dist_msk.shape[:-1], 3])
            warped_msks = np.broadcast_to(warped_msks.detach().cpu().numpy(), [*warped_msks.shape[:-1], 3])
            stylized_content = torch.movedim(torch.clamp(stylized_content, 0., 1.), 1, -1).detach().cpu().numpy()
            for i in range(warped_stylized_content0.shape[0]):
                Image.fromarray(np.uint8(255 * warped_stylized_content0[i])).save(log_dir + '/warped_stylized_content_%03d.png' % i)
                Image.fromarray(np.uint8(255 * stylized_content[i])).save(log_dir + '/stylized_content_%03d.png' % i)
                Image.fromarray(np.uint8(255 * coor_dist_msk[i])).save(log_dir + '/coor_dist_msk_%03d.png' % i)
                Image.fromarray(np.uint8(255 * warped_msks[i])).save(log_dir + '/warped_mask_%03d.png' % i)
            Image.fromarray(np.uint8(255*torch.movedim(style_images[0], 0, -1).detach().cpu().numpy())).save(log_dir + '/style_image.png')
    writer.close()


def train_temporal_invoke_pl(save_dir, sv_name, log_dir, nerf_content_dir, style_dir, batch_size, n_threads=8, lr=1e-3, max_iter=5000):

    temporal_weight, content_weight, style_weight = 100., 1.0, 1.
    print_interval = 20
    save_model_interval = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    save_dir, log_dir = str(save_dir), str(log_dir)

    decoder = VGGNet.decoder
    vgg = VGGNet.vgg

    ckpts = [os.path.join(save_dir, f) for f in sorted(os.listdir(save_dir)) if sv_name in f]
    if len(ckpts) > 0:
        ld_dict = torch.load(ckpts[-1])
        decoder.load_state_dict(ld_dict['decoder'])
        step = ld_dict['step']
    else:
        print('From original pth file')
        decoder.load_state_dict(torch.load('./pretrained/decoder.pth'))
        shutil.copy('./pretrained/decoder.pth', save_dir + '/' + sv_name)
        step = 0
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = VGGNet.Net(vgg, decoder)
    network.train()
    network.to(device)

    style_tf = train_transform2()

    content_dataset = CoorImageDataset_pl(nerf_content_dir)
    style_dataset = FlatFolderDataset(style_dir, style_tf)

    # Camera for Rendering
    h, w, focal = content_dataset.hwf
    h, w = int(h), int(w)
    cx, cy = w/2, h/2
    near_prj, far_prj = 1e-3, 1e5
    projectionMatrix = np.array([[-2*focal/w, 0,          1-2*cx/w,               0],
                                 [0,          2*focal/h,  2*cy/h-1,               0],
                                 [0,          0,          -(far_prj+near_prj)/(far_prj-near_prj), -2*far_prj*near_prj/(far_prj-near_prj)],
                                 [0,          0,          -1,                     0]])
    camera = Camera(projectionMatrix=projectionMatrix)
    camera.to(device)

    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=1,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=n_threads))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=lr)

    space_dist_threshold = 5e-2

    def adjust_learning_rate_local(optimizer, iteration_count):
        """Imitating the original implementation"""
        lr = 1e-4 / (1.0 + 5e-5 * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for i in tqdm(range(step, max_iter)):
        # Sampling Patch
        patch_size = 512
        if patch_size > 0:
            patch_h_min, patch_w_min = np.random.randint(0, h - patch_size), np.random.randint(0, w - patch_size)
            patch_h_max, patch_w_max = patch_h_min + patch_size, patch_w_min + patch_size
        else:
            patch_h_min, patch_w_min = 0, 0
            patch_h_max, patch_w_max = h, w

        resample_layer = nn.Upsample(size=(int(patch_h_max - patch_h_min), int(patch_w_max - patch_w_min)),
                                     mode='bilinear', align_corners=True)

        adjust_learning_rate_local(optimizer, iteration_count=i)
        content_images, coor_maps, cps = content_dataset.get_batch(batch_size=batch_size)
        content_images, coor_maps, cps = content_images[..., patch_h_min: patch_h_max, patch_w_min: patch_w_max].to(device),\
                                         coor_maps[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max].to(device),\
                                         cps.to(device)
        # The same style image
        style_images = next(style_iter).to(device)
        style_images = style_images[:1].expand([batch_size, * style_images.shape[1:]])

        loss_c, loss_s, stylized_content = network(content_images, style_images, return_stylized_content=True)
        stylized_content = resample_layer(stylized_content)

        # Set camera pose
        camera.set(cameraPose=cps)
        pcl_coor_world0 = coor_maps[0].reshape([-1, 3])
        pcl_rgb0 = torch.movedim(stylized_content[0], 0, -1).reshape([-1, 3])

        warped_stylized_content0, warped_coor_map0, warped_msks = camera.rasterize(pcl_coor_world0, pcl_rgb0, h=h, w=w)
        warped_stylized_content0, warped_coor_map0, warped_msks = warped_stylized_content0[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max],\
                                                                  warped_coor_map0[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max],\
                                                                  warped_msks[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max]
        coor_dist_msk = (((warped_coor_map0 - coor_maps) ** 2).sum(-1, keepdim=True) < space_dist_threshold ** 2).float()

        loss_t = (((torch.movedim(stylized_content, 1, -1) - warped_stylized_content0) ** 2) * warped_msks * coor_dist_msk).mean()
        loss_t = temporal_weight * loss_t

        loss_c = content_weight * loss_c
        loss_s = style_weight * loss_s
        loss = loss_c + loss_s + loss_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_temporal', loss_t.item(), i + 1)

        if (i + 1) % print_interval == 0:
            print('Iter %d Content Loss: %.3f Style Loss: %.3f Temporal Loss: %.3f' % (i, loss_c.item(), loss_s.item(), loss_t.item()))

        if i == 0 or (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            sv_dict = {'decoder': state_dict, 'step': (i+1)}
            torch.save(sv_dict, save_dir + '/' + sv_name)

            warped_stylized_content0 = torch.clamp(warped_stylized_content0, 0, 1).detach().cpu().numpy()
            coor_dist_msk = np.broadcast_to(coor_dist_msk.detach().cpu().numpy(), [*coor_dist_msk.shape[:-1], 3])
            warped_msks = np.broadcast_to(warped_msks.detach().cpu().numpy(), [*warped_msks.shape[:-1], 3])
            stylized_content = torch.movedim(torch.clamp(stylized_content, 0., 1.), 1, -1).detach().cpu().numpy()
            for i in range(warped_stylized_content0.shape[0]):
                Image.fromarray(np.uint8(255 * warped_stylized_content0[i])).save(log_dir + '/warped_stylized_content_%03d.png' % i)
                Image.fromarray(np.uint8(255 * stylized_content[i])).save(log_dir + '/stylized_content_%03d.png' % i)
                Image.fromarray(np.uint8(255 * coor_dist_msk[i])).save(log_dir + '/coor_dist_msk_%03d.png' % i)
                Image.fromarray(np.uint8(255 * warped_msks[i])).save(log_dir + '/warped_mask_%03d.png' % i)
            Image.fromarray(np.uint8(255*torch.movedim(style_images[0], 0, -1).detach().cpu().numpy())).save(log_dir + '/style_image.png')

    writer.close()


def write_ply_rgb(points, RGB, filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as PLY file """
    N = points.shape[0]
    vertex = []
    for i in range(N):
        vertex.append((points[i, 0], points[i, 1], points[i, 2], RGB[i][0], RGB[i][1], RGB[i][2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def ndc2world(coor_ndc, h, w, focal):
    z = 2 / (coor_ndc[..., -1] - 1)
    x = - w / 2 / focal * z * coor_ndc[..., 0]
    y = - h / 2 / focal * z * coor_ndc[..., 1]
    coor_world = torch.stack([x, y, z], dim=-1)
    return coor_world


def train_temporal_decoder(args):
    if not args.no_ndc:
        print("Using NDC Coordinate System! Check Nerf and dataset to be LLFF !!!!!!!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    decoder = VGGNet.decoder
    vgg = VGGNet.vgg

    ckpts = [os.path.join(save_dir, f) for f in sorted(os.listdir(save_dir)) if 'decoder_iter_' in f]
    if len(ckpts) > 0 and not args.no_reload:
        ld_dict = torch.load(ckpts[-1])
        decoder.load_state_dict(ld_dict['decoder'])
        step = ld_dict['step']
    else:
        print('From original pth file')
        decoder.load_state_dict(torch.load('./pretrained/decoder.pth'))
        step = 0
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = VGGNet.Net(vgg, decoder)
    network.train()
    network.to(device)

    style_tf = train_transform2()

    content_dataset = CoorImageDataset(args.nerf_content_dir)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    # Camera for Rendering
    h, w, focal = content_dataset.hwf
    h, w = int(h), int(w)
    cx, cy = w/2, h/2
    near_prj, far_prj = 1e-3, 1e5
    projectionMatrix = np.array([[-2*focal/w, 0,          1-2*cx/w,               0],
                                 [0,          2*focal/h,  2*cy/h-1,               0],
                                 [0,          0,          -(far_prj+near_prj)/(far_prj-near_prj), -2*far_prj*near_prj/(far_prj-near_prj)],
                                 [0,          0,          -1,                     0]])
    camera = Camera(projectionMatrix=projectionMatrix)
    camera.to(device)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=1,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    # Sampling Patch
    patch_size = 512
    if patch_size > 0:
        patch_h_min, patch_w_min = np.random.randint(0, h-patch_size), np.random.randint(0, w-patch_size)
        patch_h_max, patch_w_max = patch_h_min + patch_size, patch_w_min + patch_size
    else:
        patch_h_min, patch_w_min = 0, 0
        patch_h_max, patch_w_max = h, w

    resample_layer = nn.Upsample(size=(int(patch_h_max - patch_h_min), int(patch_w_max - patch_w_min)), mode='bilinear', align_corners=True)
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    space_dist_threshold = 5e-2

    for i in tqdm(range(step, args.max_iter)):

        adjust_learning_rate(optimizer, iteration_count=i)
        content_images, coor_maps, cps = next(content_iter)
        content_images, coor_maps, cps = content_images[..., patch_h_min: patch_h_max, patch_w_min: patch_w_max].to(device),\
                                         coor_maps[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max].to(device),\
                                         cps.to(device)
        if not args.no_ndc:
            coor_maps = ndc2world(coor_maps, h, w, focal)

        # The same style image
        style_images = next(style_iter).to(device)
        style_images = style_images[:1].expand([args.batch_size, * style_images.shape[1:]])

        loss_c, loss_s, stylized_content = network(content_images, style_images, return_stylized_content=True)
        stylized_content = resample_layer(stylized_content)

        # Set camera pose
        camera.set(cameraPose=cps)
        pcl_coor_world0 = coor_maps[0].reshape([-1, 3])
        pcl_rgb0 = torch.movedim(stylized_content[0], 0, -1).reshape([-1, 3])

        warped_stylized_content0, warped_coor_map0, warped_msks = camera.rasterize(pcl_coor_world0, pcl_rgb0, h=h, w=w)
        warped_stylized_content0, warped_coor_map0, warped_msks = warped_stylized_content0[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max],\
                                                                  warped_coor_map0[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max],\
                                                                  warped_msks[:, patch_h_min: patch_h_max, patch_w_min: patch_w_max]
        coor_dist_msk = (((warped_coor_map0 - coor_maps) ** 2).sum(-1, keepdim=True) < space_dist_threshold ** 2).float()

        loss_t = (((torch.movedim(stylized_content, 1, -1) - warped_stylized_content0) ** 2) * warped_msks * coor_dist_msk).mean()
        loss_t = args.temporal_weight * loss_t

        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s + loss_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_temporal', loss_t.item(), i + 1)

        if (i + 1) % args.print_interval == 0:
            print('Iter %d Content Loss: %.3f Style Loss: %.3f Temporal Loss: %.3f' % (i, loss_c.item(), loss_s.item(), loss_t.item()))

        if i == 0 or (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            sv_dict = {'decoder': state_dict, 'step': (i+1)}
            torch.save(sv_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))
            # Delete ckpts
            ckpts = [os.path.join(save_dir, f) for f in sorted(os.listdir(save_dir)) if 'decoder_iter_' in f]
            if len(ckpts) > args.ckp_num:
                os.remove(ckpts[0])

            warped_stylized_content0 = torch.clamp(warped_stylized_content0, 0, 1).detach().cpu().numpy()
            coor_dist_msk = np.broadcast_to(coor_dist_msk.detach().cpu().numpy(), [*coor_dist_msk.shape[:-1], 3])
            warped_msks = np.broadcast_to(warped_msks.detach().cpu().numpy(), [*warped_msks.shape[:-1], 3])
            stylized_content = torch.movedim(torch.clamp(stylized_content, 0., 1.), 1, -1).detach().cpu().numpy()
            for i in range(warped_stylized_content0.shape[0]):
                Image.fromarray(np.uint8(255 * warped_stylized_content0[i])).save(args.log_dir + '/warped_stylized_content_%03d.png' % i)
                Image.fromarray(np.uint8(255 * stylized_content[i])).save(args.log_dir + '/stylized_content_%03d.png' % i)
                Image.fromarray(np.uint8(255 * coor_dist_msk[i])).save(args.log_dir + '/coor_dist_msk_%03d.png' % i)
                Image.fromarray(np.uint8(255 * warped_msks[i])).save(args.log_dir + '/warped_mask_%03d.png' % i)
            Image.fromarray(np.uint8(255*torch.movedim(style_images[0], 0, -1).detach().cpu().numpy())).save(args.log_dir + '/style_image.png')

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vae',
                        help='vae or finetune_decoder')
    # Basic options
    parser.add_argument('--content_dir', type=str, default='./all_contents/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--nerf_content_dir', type=str, default='./nerf_gen_data2/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='./all_styles/',
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='./pretrained/vgg_normalised.pth')

    parser.add_argument('--no_ndc', action='store_true')
    parser.add_argument('--no_reload', action='store_true')

    # training options
    parser.add_argument('--save_dir', default='./pretrained/',
                        help='Directory to save the model')
    parser.add_argument('--ckp_num', type=int, default=3)
    parser.add_argument('--log_dir', default='./logs/stylenet/',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=2.)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--temporal_weight', type=float, default=50.)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=200)
    parser.add_argument('--print_interval', type=int, default=20)

    # train vae options
    parser.add_argument('--vae_d', type=int, default=4)
    parser.add_argument('--vae_w', type=int, default=512)
    parser.add_argument('--vae_latent', type=int, default=32)
    parser.add_argument('--vae_kl_lambda', type=float, default=0.1)
    args = parser.parse_args()

    if args.task == 'finetune_decoder':
        finetune_decoder(args)
    elif args.task == 'vae':
        train_vae(args)
    else:
        train_temporal_decoder()
