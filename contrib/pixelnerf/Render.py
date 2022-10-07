import jittor as jt

def render_rays(net, rays, bound, N_samples, ref, noise_std=.0):
    rays_o, rays_d = rays
    near, far = bound
    N_c, N_f = N_samples

    # coarse sampling
    z_vals = get_coarse_query_points(near, far, N_c)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    # get projection feature
    f_projection = ref.feature_matching(pts)
    # neural rendering
    rgb, w = get_rgb_w(net, pts, rays_d, z_vals, f_projection, noise_std)
    rgb_map = jt.sum(w[..., None] * rgb, dim=-2)
    depth_map = jt.sum(w * z_vals, -1)
    acc_map = jt.sum(w, -1)
    return rgb_map, depth_map, acc_map


def get_coarse_query_points(tn, tf, N_samples):
    k = jt.rand([N_samples]) / float(N_samples)
    pt_value = jt.linspace(0.0, 1.0, N_samples + 1)[:-1]
    pt_value += k
    return tn + (tf - tn) * pt_value


def get_rgb_w(net, pts, rays_d, z_vals, ref_feature, noise_std=.0):
    rgb, sigma = net(ref_feature, pts, rays_d)
    rgb = rgb.view(list(pts.shape[:-1]) + [3])
    sigma = sigma.view(list(pts.shape[:-1]))
    # get the interval
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    INF = jt.ones(delta[..., :1].shape).masked_fill(jt.ones(delta[..., :1].shape), 1e10)
    delta = jt.concat([delta, INF], -1)
    delta = delta * jt.norm(rays_d, dim=-1, keepdim=True)

    # add noise to sigma
    if noise_std > 0.:
        sigma += jt.randn(sigma.size()) * noise_std

    # get weights
    alpha = 1. - jt.exp(-sigma * delta)
    ones = jt.ones(alpha[..., :1].shape)
    weights = alpha * jt.cumprod(jt.concat([ones, 1. - alpha + 1e-7], dim=-1), dim=-1)[..., :-1]

    return rgb, weights
