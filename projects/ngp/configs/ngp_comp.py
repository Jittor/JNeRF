sampler = dict(
    type='DensityGridSampler',
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

exp_name = "Scar"
dataset_type = 'NerfDataset'
dataset_dir = 'my/data/'+exp_name
dataset_aabb = {"Car":4, "Coffee":1, "Easyship":8, "Scar":5, "Scarf":8}
dataset_scale = {"Car":None, "Coffee":None, "Easyship":None, "Scar":None, "Scarf":0.05}
dataset_offset = {"Car":[-2.0, -0.5, 0.0], "Coffee":None, "Easyship":None, "Scar":None, "Scarf":None}

dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1,-1,1],
        batch_size=4096,
        mode='train',
        aabb_scale=dataset_aabb[exp_name],
        scale=dataset_scale[exp_name],
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
        scale=dataset_scale[exp_name],
        offset=dataset_offset[exp_name],
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1,-1,1],
        batch_size=4096,
        mode='test',
        have_img=False,
        H=800,
        W=800,
        preload_shuffle=False,
        aabb_scale=dataset_aabb[exp_name],
        scale=dataset_scale[exp_name],
        offset=dataset_offset[exp_name],
    ),
)

log_dir = "./logs"
tot_train_steps = 40000
# Background color, value range from 0 to 1
background_color = [1, 1, 1]
# Hash encoding function used in Instant-NGP
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
# Expected number of sampling points per batch
target_batch_size = 1<<18
# Set const_dt=True for higher performance
# Set const_dt=False for faster convergence
const_dt=True
# Use fp16 for faster training
fp16 = True
