import jittor as jt
import numpy as np
import logging
import mcubes

import time, functools

def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = fn(*args, **kwargs)
        print('%s executed in %s s' % (fn.__name__, (time.time() - t1)))
        return res
    return wrapper


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = jt.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = jt.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = jt.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with jt.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = jt.meshgrid(xs, ys, zs)
                    pts = jt.concat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, do_dilation=False):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    if do_dilation:
        import scipy
        kernel_size = 5
        print("do the density dilation with kernel size %d ! " % (kernel_size))
        # erosion since the sdf outside is negative
        u = scipy.ndimage.morphology.grey_dilation(u, size=(kernel_size,kernel_size,kernel_size))
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.numpy()
    b_min_np = bound_min.numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = jt.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds - 1), inds - 1)
    above = jt.minimum((cdf.shape[-1] - 1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = jt.where(denom < 1e-5, jt.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    # @metric
    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.array([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = jt.linalg.norm(pts, ord=2, dim=-1, keepdim=True).safe_clip(1.0, 1e10)
        pts = jt.concat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - jt.exp(-jt.nn.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, pts):
        pass

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, pts, last=False):
        pass

    # @metric
    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    pts,
                    view_dir,
                    setZero=None,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.array([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        # pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        if view_dir != None:
            dirs = view_dir[:, None, :].expand(pts.shape)
        else:
            dirs = rays_d[:, None, :].expand(pts.shape)

        ### only consider the valid points
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts)

        with jt.no_grad():    # if setZero != None:
            if isinstance(setZero, jt.Var):
                # pass
                sdf[setZero.reshape(-1)] = 10000
                feature_vector[setZero.reshape(-1)] = 0
                gradients[setZero.reshape(-1)] = 0
            sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

            inv_s = deviation_network(jt.zeros([1, 3]))[:, :1].safe_clip(1e-6, 1e6)           # Single parameter
            inv_s = inv_s.expand(batch_size * n_samples, 1)

            true_cos = (dirs * gradients).sum(-1, keepdims=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(jt.nn.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        jt.nn.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = jt.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = jt.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).safe_clip(0.0, 1.0)

            pts_norm = jt.norm(pts, p=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
            # inside_sphere = (pts_norm < 1.0).float().detach()
            inside_sphere = (pts_norm < 1.5).float().detach()

            relax_inside_sphere = (pts_norm < 1.2).float().detach()

            # Render with background
            if background_alpha is not None:
                alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
                alpha = jt.concat([alpha, background_alpha[:, n_samples:]], dim=-1)
                sampled_color = sampled_color * inside_sphere[:, :, None] +\
                                background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
                sampled_color = jt.concat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

            weights = alpha * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

            # set zero
            # if setZero != None:
            if isinstance(setZero, jt.Var):
                weights[setZero.squeeze(-1)==True] = 0

            weights_sum = weights.sum(dim=-1, keepdims=True)

            color = (sampled_color * weights[:, :, None]).sum(dim=1)
            if background_rgb is not None:    # Fixed background, usually black
                color = color + background_rgb * (1.0 - weights_sum)

            # Eikonal loss
            gradient_error = (jt.norm(gradients.reshape(batch_size, n_samples, 3), p=2,
                                                dim=-1) - 1.0) ** 2
            gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    # @metric
    def render(self, rays_o, rays_d, near, far, view_dir, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0,
                    use_deform=False, query_delta=None, hull=None, deltas=None, vis_coord_ind=-1):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = jt.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = jt.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        background_alpha = None
        background_sampled_color = None

        # Background model
        if self.n_outside > 0:
            z_vals_feed = jt.concat([z_vals, z_vals_outside], dim=-1)
            _, z_vals_feed = jt.argsort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.array([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

        if vis_coord_ind >= 0 and vis_coord_ind < pts.shape[0]: # visualize ray
            pickedRays = pts[vis_coord_ind] # [numSample 3]
            save_path = './vis_ray_ori.obj'
            with open(save_path, 'w') as f:
                for pt in pickedRays.cpu().numpy(): 
                    f.write("v %s %s %s\n" % (pt[0], pt[1], pt[2]))
                print("ray has visualized")

        if use_deform:
            det, tri_verts = query_delta(hull, deltas, pts)
            pts += det
            setZero = tri_verts[-1]
        else:
            setZero = None

        if vis_coord_ind >= 0 and vis_coord_ind < pts.shape[0]: # visualize ray
            pickedRays = pts[vis_coord_ind] # [numSample 3]
            save_path = './vis_ray.obj'
            with open(save_path, 'w') as f:
                for pt in pickedRays.cpu().numpy(): 
                    f.write("v %s %s %s\n" % (pt[0], pt[1], pt[2]))
                print("ray has visualized")

        # Render core
        # import time
        # t1 = time.time()
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    pts,
                                    view_dir,
                                    setZero,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)
        # print("rendering batch core cost time %s" % (time.time()-t1))
        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdims=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdims=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': jt.max(weights, dim=-1, keepdims=True),
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0, do_dilation=False):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts), do_dilation=do_dilation)
