model = dict(
    type='NeuS',
)
optim = dict(
    type='Adam',
    lr=1e-2,
    eps=1e-15,
    betas=(0.9,0.99),
)

dataset = dict(
    type = 'NeuSDataset',
    dataset_dir = '/data/yqs/mesh_recon/NeuS_jittor/data_thin_structure/thin_cube',
    render_cameras_name = 'cameras_sphere.npz',
    object_cameras_name = 'cameras_sphere.npz',
)

model = dict(
    type = 'NeuS',
    nerf_network = dict(
        D = 8,
        d_in = 4,
        d_in_view = 3,
        W = 256,
        multires = 10,
        multires_view = 4,
        output_ch = 4,
        skips=[4],
        use_viewdirs=True 
    ),
    sdf_network = dict(
        d_out = 257,
        d_in = 3,
        d_hidden = 256,
        n_layers = 8,
        skip_in = [4],
        multires = 6,
        bias = 0.5,
        scale = 1.0,
        geometric_init = True,
        weight_norm = True,       
    ),
    variance_network = dict(
        init_val = 0.3,
    ),
    rendering_network = dict(
        d_feature = 256,
        mode = 'idr',
        d_in = 9,
        d_out = 3,
        d_hidden = 256,
        n_layers = 4,
        weight_norm = True,
        multires_view = 4,
        squeeze_out = True,
    ), 
)

render = dict(
    type = 'NeuSRenderer',
    n_samples = 64,
    n_importance = 64,
    n_outside = 32,
    up_sample_steps = 4,     # 1 for simple coarse-to-fine sampling
    perturb = 1.0,
)

base_exp_dir = './log/thin_cube/womask'
recording = [ './','./models']

learning_rate = 5e-4
learning_rate_alpha = 0.05
end_iter = 300000

batch_size = 512
validate_resolution_level = 6
warm_up_end = 5000
anneal_end = 50000
use_white_bkgd = False

save_freq = 10000
val_freq = 200
val_mesh_freq = 5000
report_freq = 100

igr_weight = 0.1
mask_weight = 0.0