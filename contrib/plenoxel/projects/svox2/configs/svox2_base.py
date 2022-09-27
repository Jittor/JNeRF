
exp_name = "lego"
log_dir = "./logs"
tot_train_steps = 40000
# Background color, value range from 0 to 1
background_color = [0, 0, 0]
fp16 = True
# Load pre-trained model
load_ckpt = False
# path of checkpoint file, None for default path
ckpt_path = None
# test output image with alpha
alpha_image = False

reso_list = [[256]*3, [512]*3]
epoch_size = 12800
batch_size = 5000

lr_basis = 1e-06
lr_basis_begin_step = 0
lr_basis_decay_steps = 250000
lr_basis_delay_mult = 0.01
lr_basis_delay_steps = 0
lr_basis_final = 1e-06
lr_color_bg = 0.1
lr_color_bg_decay_steps = 250000
lr_color_bg_delay_mult = 0.01
lr_color_bg_delay_steps = 0
lr_color_bg_final = 5e-06
lr_decay = True
lr_fg_begin_step = 0
lr_sh = 0.01
lr_sh_decay_steps = 250000
lr_sh_delay_mult = 0.01
lr_sh_delay_steps = 0
lr_sh_final = 5e-06
lr_sigma = 30.0
lr_sigma_bg = 3.0
lr_sigma_bg_decay_steps = 250000
lr_sigma_bg_delay_mult = 0.01
lr_sigma_bg_delay_steps = 0
lr_sigma_bg_final = 0.003
lr_sigma_decay_steps = 250000
lr_sigma_delay_mult = 0.01
lr_sigma_delay_steps = 15000
lr_sigma_final = 0.05


lambda_tv = 1e-05
lambda_tv_sh = 0.001
tv_contiguous = 1
tv_sh_sparsity = 0.01
tv_logalpha = False
tv_sparsity = 0.01
eval_every = 1
print_every = 20
init_sigma = 0.1
init_sigma_bg = 0.1
sigma_thresh = 1e-08
step_size = 0.5
stop_thresh = 1e-07
background_brightness = 1.0
random_sigma_std = 0.0
random_sigma_std_background = 0.0
last_sample_opaque = False
near_clip = 0.0
use_spheric_clip = False
init_iters = 0
enable_random = False
rms_beta = 0.95
weight_decay_sh = 1.0
weight_decay_sigma = 1.0
upsamp_every = 38400
tv_early_only = 1
tv_decay = 1.0
density_thresh = 5.0
weight_thresh = 0.256
thresh_type = 'weight'
max_grid_elements = 44000000
upsample_density_add = 0.0
n_iters=128000
model = dict(
    type='SparseGrid',
    basis_dim=9,
    basis_reso=32,
    nosphereinit=False,
)


dataset_type = 'SvoxNeRFDataset'
dataset_dir = 'data/lego'
dataset = dict(
    train=dict(
        type=dataset_type,
        root=dataset_dir,
        split='train',
        epoch_size=epoch_size*batch_size,
    ),
    test=dict(
        type=dataset_type,
        root=dataset_dir,
        split='test',
        epoch_size=epoch_size*batch_size,
    ),
)

loss = dict(
    type='MSELoss'
)
