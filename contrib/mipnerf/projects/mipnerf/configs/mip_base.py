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
    lr=8e-3,
    eps=1e-15,
    betas=(0.9, 0.99),
)

dataset_type = 'Blender'
dataset_dir = "nerf_data/nerf_synthetic/lego/" 
dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=288,
        mode='train',
    ),
    val=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode='val',
        preload_shuffle=False,
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode='test',
        preload_shuffle=False,
    ),
)

exp_name = "lego_sss"
log_dir = "./logs"
tot_train_steps = 40001
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
num_levels = 2
num_samples = 128
net_depth = 8
skip_layer = 4
net_width = 256
net_depth_condition = 1
net_width_condition = 128
num_density_channels = 1
num_rgb_channels = 3
resample_padding = 0.01
lindisp = False
ray_shape = 'cone'
min_deg_point = 0
max_deg_point = 8
coarse_loss_mult = 0.1
disable_multiscale_loss = False
randomized = True
disable_integration = False
use_viewdirs = True
deg_view = 4
density_noise = 0.
density_bias = -1.
rgb_padding = 0.001
stop_level_grad = True
near = 2.
far = 6.
linearlog = dict(
    type='LinearLog',
    end_lr=5e-6,
    max_steps=tot_train_steps,
    lr_delay_steps=2500,
    lr_delay_mult=0.01,
)
