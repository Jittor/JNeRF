sampler = dict(
    type='MipSampler',
)
model = dict(
    type='MipNerfMLP',
)
loss = dict(
    type='MSELoss',
)
optim = dict(
    type='Adam',
    lr=5e-4,
    eps=1e-15,
    betas=(0.9, 0.99),
)

dataset_type = 'Blenders'
dataset_dir = "/mnt/disk3/zhm/data/nerf_synthetic/lego/"  # 这个仓库的相对路径需要加上 .
dataset = dict(
    train=dict(
        type=dataset_type,
        data_dir=dataset_dir,
        batch_size=3072,
        split='train',
        batch_type='all_images'
    ),
    val=dict(
        type=dataset_type,
        data_dir=dataset_dir,
        batch_size=1,
        split='val',
        batch_type='single_image'
    ),
    test=dict(
        type=dataset_type,
        data_dir=dataset_dir,
        batch_size=1,
        split='test',
        batch_type='single_image'
    ),
)

# nerf
num_levels = 2  # sampling levels
num_samples = 128  # the number of samples per level
resample_padding = 0.01  # Dirichlet/alpha "padding" on the histogram.
rgb_padding = 0.001  # Padding added to the RGB outputs.
# MLP
net_depth = 8  # net_depth
skip_layer = 4  # skip_depth
net_width = 256  #
net_depth_condition = 1  # The depth of the second part of MLP.
net_width_condition = 128  # The width of the second part of MLP.
num_density_channels = 1  #
num_rgb_channels = 3
ray_shape = 'cone'
min_deg_point = 0   # Min degree of positional encoding for 3D points.
max_deg_point = 16   # pl中使用16， 浩洋师兄是8
use_viewdirs = True
deg_view = 4   # Degree of positional encoding for viewdirs.
density_noise = 0.  # Standard deviation of noise added to raw density
density_bias = -1.  # The shift added to raw densities pre-activation.


exp_name = "test"
log_dir = "./logs"
tot_train_steps = 120001
background_color = [0, 0, 0]
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
target_batch_size = 1 << 18
const_dt = True
fp16 = False
white_bkgd = False
using_fp16 = False
lindisp = False

# LOSS
coarse_loss_mult = 0.1
disable_multiscale_loss = False
randomized = True
disable_integration = False


stop_level_grad = True
near = 2.
far = 6.

# optimizer
linearlog = dict(
    type='LinearLog',
    end_lr=5e-6,
    max_steps=tot_train_steps,
    lr_delay_steps=2500,
    lr_delay_mult=0.01,
)
