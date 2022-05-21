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

exp_name = "Easyship"
dataset_type = 'NerfDataset'
dataset_dir = 'my/data/'+exp_name
dataset_aabb = {"Car":4, "Coffee":1, "Easyship":5, "Scar":5, "Scarf":13}
dataset_offset = {"Car":[-2.0, -0.5, 0.0], "Coffee":None, "Easyship":None, "Scar":None, "Scarf":None}

dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1,-1,1],
        batch_size=4096,
        mode='train',
        aabb_scale=dataset_aabb[exp_name],
        offset=dataset_offset[exp_name],
    ),
    val=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1,-1,1],
        batch_size=4096,
        mode='val',
        preload_shuffle=False,
        aabb_scale=dataset_aabb[exp_name],
        offset=dataset_offset[exp_name],
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1,-1,1],
        batch_size=4096,
        mode='test',
        # have_img=False,
        H=800,
        W=800,
        preload_shuffle=False,
        aabb_scale=dataset_aabb[exp_name],
        offset=dataset_offset[exp_name],
    ),
)

log_dir = "./logs"
tot_train_steps = 40000
background_color = [1, 1, 1]
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
target_batch_size = 1<<18
const_dt=True
fp16 = True
