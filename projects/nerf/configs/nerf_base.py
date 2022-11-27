sampler = dict(
    type='DensityGridSampler',
    update_den_freq=16,
)
encoder = dict(
    pos_encoder = dict(
        type='FrequencyEncoder',
        multires=10,
    ),
    dir_encoder = dict(
        type='FrequencyEncoder',
        multires=4,
    ),
)
model = dict(
    type='OriginNeRFNetworks',
)
loss = dict(
    type='HuberLoss',
    delta=0.1,
)
optim = dict(
    type='Adam',
    lr=1e-2,
    eps=1e-15,
    betas=(0.9,0.99),
)
ema = dict(
    type='EMA',
    decay=0.95,
)
expdecay=dict(
    type='ExpDecay',
    decay_start=20_000,
    decay_interval=10_000,
    decay_base=0.33,
    decay_end=None
)
dataset_type = 'NerfDataset'
dataset_dir = 'data/lego'
dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=512,
        mode='train',
    ),
    val=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=512,
        mode='val',
        preload_shuffle=False,
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=512,
        mode='test',
        preload_shuffle=False,
    ),
)

exp_name = "lego"
log_dir = "./logs"
tot_train_steps = 200000
# Background color, value range from 0 to 1
background_color = [0, 0, 0]
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 512
n_training_steps = 16
# Expected number of sampling points per batch
target_batch_size = 1<<18
# Set const_dt=True for higher performance
# Set const_dt=False for faster convergence
const_dt=True
# Use fp16 for faster training
fp16 = True
