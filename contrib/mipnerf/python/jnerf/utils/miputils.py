import collections
import os
import numpy as np
import jittor as jt

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


def clip(x, down, up):
    assert x.shape[0] == up.shape[0]
    for i in range(x.shape[0]):
        x[i] = jt.maximum(x[i], down[i])
        x[i] = jt.minimum(x[i], up[i])
    return x


class finfo:
    def __init__(self, kind):
        self.kind = kind

    def eps(self):
        if self.kind == "float32":
            return np.finfo("float32").eps
        elif self.kind == "float16":
            return np.finfo("float16").eps

    def max(self):
        if self.kind == "float32":
            return np.finfo("float32").max
        elif self.kind == "float16":
            return np.finfo("float16").max

    def min(self):
        if self.kind == "float32":
            return np.finfo("float32").min
        elif self.kind == "float16":
            return np.finfo("float16").min


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if posinf is None:
        posinf = np.finfo(str(x.dtype)).max
    if neginf is None:
        neginf = np.finfo(str(x.dtype)).min
    tnan = np.array((float("nan"))).astype(str(x.dtype))
    inf = np.array(np.float16(float("inf"))).astype(str(x.dtype))
    minf = np.array(np.float16(float("-inf"))).astype(str(x.dtype))
    # x[x == tnan] = nan
    # x[x == inf] = posinf
    # x[x == minf] = neginf
    return x


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling from sorted bins.
    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = jt.sum(weights, dim=-1, keepdims=True)

    padding = jt.maximum(0, eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = jt.minimum(1, jt.cumsum(pdf[..., :-1], dim=-1))
    cdf = jt.concat([
        jt.zeros(list(cdf.shape[:-1]) + [1]), cdf,
        jt.ones(list(cdf.shape[:-1]) + [1])
    ], dim=-1)
    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = jt.arange(num_samples) * s
        u += jt.array(np.random.uniform(0, s - finfo(str(u.dtype)).eps(), list(cdf.shape[:-1]) + [num_samples])) # random no autograd. No effects here.
        # maxval=s - jnp.finfo('float32').eps)
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = jt.minimum(u, 1. - finfo(str(u.dtype)).eps())
    else:
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = jt.linspace(0., 1. - finfo(str(cdf.dtype)).eps(), num_samples)
        u = jt.broadcast(u, list(cdf.shape[:-1]) + [num_samples])
    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = jt.max(jt.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = jt.min(jt.where(mask.logical_not(), x[..., None], x[..., -1:, None]), -2)
        # print("x0: ", mask)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)
    t = jt.safe_clip(nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples


def pos_enc(x, min_deg, max_deg, append_identity=True, using_fp16=False):
    scales = jt.array([2 ** i for i in range(min_deg, max_deg)])
    xb = (x[..., None, :] * scales[:, None]).reshape(x.shape[:-1] + [-1])
    four_feat = jt.sin(jt.concat([xb, xb + 0.5 * jt.array(np.pi)], dim=-1))
    if append_identity:
        return jt.concat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def expected_sin(x, x_var):
    # When the variance is wide, shrink sin towards zero.
    y = jt.exp(-0.5 * x_var) * jt.sin(x)
    y_var = jt.maximum(
        0, 0.5 * (1 - jt.exp(-2 * x_var) * jt.cos(2 * x)) - y ** 2)
    return y, y_var


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = jt.maximum(1e-10, jt.sum(d ** 2, dim=-1, keepdims=True))
    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = jt.eye(d.shape[-1])
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.

    Args:
        d: jnp.float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).

    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                            (3 * mu ** 2 + hw ** 2) ** 2)
        r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                    (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    else:
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.

    Args:
        d: jnp.float32 3-vector, the axis of the cylinder
        t0: float, the starting distance of the cylinder.
        t1: float, the ending distance of the cylinder.
        radius: float, the radius of the cylinder
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
        a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
        t_vals: float array, the "fencepost" distances along the ray.
        origins: float array, the ray origin coordinates.
        directions: float array, the ray direction vectors.
        radii: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diag: boolean, whether or not the covariance matrices should be diagonal.

    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Args:
        x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
        be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diag: bool, if true, expects input covariances to be diagonal (full
        otherwise).

    Returns:
        encoded: jnp.ndarray, encoded variables.
    """
    if diag:
        x, x_cov_diag = x_coord
        scales = jt.array([2 ** i for i in range(min_deg, max_deg)])
        shape = list(x.shape[:-1]) + [-1]
        # jt.sync_all()
        y = (x[..., None, :] * scales[:, None]).reshape(shape)
        # jt.sync_all()
        y_var = (x_cov_diag[..., None, :] * scales[:, None] ** 2).reshape(shape)
    else:
        x, x_cov = x_coord
        num_dims = x.shape[-1]
        basis = jt.concat(
            [2 ** i * jt.eye(num_dims) for i in range(min_deg, max_deg)], 1)
        y = jt.matmul(x, basis)
        # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
        # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
        y_var = jt.sum((jt.matmul(x_cov, basis)) * basis, -2)
    return expected_sin(
        jt.concat([y, y + 0.5 * jt.array(np.pi)], dim=-1),
        jt.concat([y_var] * 2, dim=-1))[0]


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.

    Args:
        rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
        density: jnp.ndarray(float32), density, [batch_size, num_samples, 1].
        t_vals: jnp.ndarray(float32), [batch_size, num_samples].
        dirs: jnp.ndarray(float32), [batch_size, 3].
        white_bkgd: bool.

    Returns:
        comp_rgb: jnp.ndarray(float32), [batch_size, 3].
        disp: jnp.ndarray(float32), [batch_size].
        acc: jnp.ndarray(float32), [batch_size].
        weights: jnp.ndarray(float32), [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * jt.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    # jt.sync_all()
    density_delta = density[..., 0] * delta

    alpha = 1 - jt.exp(-density_delta)

    trans = jt.exp(-jt.concat([
        jt.zeros_like(density_delta[..., :1]),
        jt.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    # print(density, trans)
    weights = alpha * trans
    # jt.sync_all()

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    # jt.sync_all()
    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    # jt.sync_all()
    distance = clip(
        nan_to_num(distance, jt.array(np.inf)), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    # jt.sync_all()
    return comp_rgb, distance, acc, weights


def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, lindisp, ray_shape):
    """Stratified sampling along the rays.

    Args:
        key: jnp.ndarray, random generator key.
        origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
        radii: jnp.ndarray(float32), [batch_size, 3], ray radii.
        num_samples: int.
        near: jnp.ndarray, [batch_size, 1], near clip.
        far: jnp.ndarray, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        lindisp: bool, sampling linearly in disparity rather than depth.
        ray_shape: string, which shape ray to assume.

    Returns:
        t_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
        means: jnp.ndarray, [batch_size, num_samples, 3], sampled means.
        covs: jnp.ndarray, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]
    t_vals = jt.linspace(0., 1., num_samples + 1)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        # t_vals = near * (1. - t_vals) + far * t_vals
        t_vals = near + (far - near) * t_vals
    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = jt.concat([mids, t_vals[..., -1:]], -1)
        lower = jt.concat([t_vals[..., :1], mids], -1)
        t_rand = jt.array(np.random.uniform(0, 1, [batch_size, num_samples + 1]))  #
        # print(t_rand.unsqueeze(1).unsqueeze(1).repeat(1, upper.shape[1], upper.shape[2], 1).shape)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = t_vals.broadcast([batch_size, num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
    return t_vals, (means, covs)


def resample_along_rays(origins, directions, radii, t_vals, weights,
                        randomized, ray_shape, stop_grad, resample_padding):
    """Resampling.

    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
        radii: jnp.ndarray(float32), [batch_size, 3], ray radii.
        t_vals: jnp.ndarray(float32), [batch_size, num_samples+1].
        weights: jnp.array(float32), weights for t_vals
        randomized: bool, use randomized samples.
        ray_shape: string, which kind of shape to assume for the ray.
        stop_grad: bool, whether or not to backprop through sampling.
        resample_padding: float, added to the weights before normalizing.

    Returns:
        t_vals: jnp.ndarray(float32), [batch_size, num_samples+1].
        points: jnp.ndarray(float32), [batch_size, num_samples, 3].
    """
    # Do a blurpool.
    weights_pad = jt.concat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ], dim=-1)

    weights_max = jt.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
    # Add in a constant (the sampling function will renormalize the PDF).
    weights = weights_blur + resample_padding

    new_t_vals = sorted_piecewise_constant_pdf(
        t_vals,
        weights,
        t_vals.shape[-1],
        randomized,
    )

    if stop_grad:
        new_t_vals.requires_grad = False
        # new_t_vals = lax.stop_gradient(new_t_vals)
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions
