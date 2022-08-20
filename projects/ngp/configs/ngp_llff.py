sampler = dict(
    type='DensityGridSampler',
    update_den_freq=16,
)
encoder = dict(
    pos_encoder=dict(
        type='HashEncoder',
    ),
    dir_encoder=dict(
        type='SHEncoder',
    ),
)
model = dict(
    type='NGPNetworks',
    use_fully=True,
)
loss = dict(
    type='HuberLoss',
    delta=0.1,
)
optim = dict(
    type='Adam',
    lr=1e-1,
    eps=1e-15,
    betas=(0.9, 0.99),
)
ema = dict(
    type='EMA',
    decay=0.95,
)
expdecay = dict(
    type='ExpDecay',
    decay_start=20_000,
    decay_interval=10_000,
    decay_base=0.33,
    decay_end=None
)
dataset_type = 'LLFFDataset'
dataset_dir = 'data/fern'
dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode='train',
        factor=8,
        llffhold=8,
        aabb_scale=64,
    ),
    val=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode='val',
        preload_shuffle=False,
        factor=8,
        llffhold=8,
        aabb_scale=64,
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode='test',
        preload_shuffle=False,
        factor=8,
        llffhold=8,
        aabb_scale=64,
    ),
)

exp_name = "fern"
log_dir = "./logs"
tot_train_steps = 40000
# Background color, value range from 0 to 1
background_color = [0, 0, 0]
# Hash encoding function used in Instant-NGP
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
# Expected number of sampling points per batch
target_batch_size = 1 << 18
# Set const_dt=True for higher performance
# Set const_dt=False for faster convergence
const_dt = False
# Use fp16 for faster training
fp16 = True
# Load pre-trained model
load_ckpt = False
# path of checkpoint file, None for default path
ckpt_path = None
# test output image with alpha
alpha_image = False
