import jittor as jt
from tqdm import tqdm
from Dataset import get_dataset
from jittor.dataset import Dataset
from Network import PixelNeRF
from Render import render_rays
from test_utils import generate_video_nearby
import numpy as np
from jittor_utils import auto_diff
jt.flags.use_cuda = 1
jt.set_global_seed(999)
n_train = 3

# create rays for batch train
print("Process rays data for training!")
rays_dataset, ref_dataset = get_dataset("./tiny_nerf_data.npz", n_train)

# training parameters
Batch_size = 2048
rays_dataset.set_attrs(batch_size=Batch_size, drop_last=True, shuffle=True)
# rays_loader = Dataset(rays_dataset, batch_size=Batch_size, drop_last=True, shuffle=True)
print(f"Batch size of rays: {Batch_size}")

bound = (2., 6.)
N_samples = (64, None)
epoch = 100
img_f_ch = 512
lr = 1e-4

# training
net = PixelNeRF(img_f_ch)

mse = jt.nn.MSELoss()
optimizer = jt.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)
net.load("/home/penghy/torch_pixel/tt.pth")
print("Start Training!")
for e in range(epoch):
    print(len(rays_dataset))
    with tqdm(total=len(rays_dataset) // Batch_size, desc=f"Epoch {e+1}", ncols=100) as p_bar:
        for it, train_rays in enumerate(rays_dataset):
            assert train_rays.shape == (Batch_size, 9)
            rays_o, rays_d, target_rgb = jt.chunk(train_rays, 3, dim=-1)
            rays_od = (rays_o, rays_d)
            rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, ref=ref_dataset)
            loss = mse(rgb, target_rgb)
            optimizer.step(loss)
            jt.sync_all()
            p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
            p_bar.update(1)

print('Finish Training!')

print('Start Generating Video!')
net.eval()
generate_video_nearby(net, ref_dataset, bound, N_samples, './video/test.mp4')
print('Finish Generating Video!')



