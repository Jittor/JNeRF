sampler = dict(
    type='DensityGirdSampler',
    update_den_freq=16,
)
encoder = dict(
    pos_encoder = dict(
        type='HashEncoder',
    ),
    dir_encoder = dict(
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
dataset_dir = 'my/data/lego'
dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
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

exp_name = "lego"
log_dir = "./logs"
tot_train_steps = 40000
background_color = [0, 0, 0]
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
target_batch_size = 1<<18
const_dt=True
fp16 = True
