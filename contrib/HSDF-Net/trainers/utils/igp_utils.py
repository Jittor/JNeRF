#import torch
import jittor
from trainers.utils.diff_ops import gradient, jacobian


def outter(v1, v2):
    """
    Batched outter product of two vectors: [v1] [v2]^T
    :param v1: (bs, dim)
    :param v2: (bs, dim)
    :return: (bs, dim, dim)
    """
    bs = v1.size(0)
    d = v1.size(1)
    v1 = v1.view(bs, d, 1)
    v2 = v2.view(bs, 1, d)
    return jittor.bmm(v1, v2)


def _addr_(mat, vec1, vec2, alpha=1., beta=1.):
    """
    Return
        alpha * outter(vec1, vec2) + beta * [mat]
    :param mat:  (bs, npoints, dim, dim)
    :param vec1: (bs, npoints, dim)
    :param vec2: (bs, npoints, dim)
    :param alpha: float
    :param beta: float
    :return:
    """
    bs, npoints, dim =vec1.size(0), vec1.size(1), vec1.size(2)
    assert len(mat.size()) == 4
    outter_n = outter(vec1.view(bs * npoints, dim), vec2.view(bs * npoints, dim))
    outter_n = outter_n.view(bs, npoints, dim, dim)
    out = alpha * outter_n + beta * mat.view(bs, npoints, dim, dim)
    return out


def get_surf_pcl(net, npoints=1000, dim=3, use_rejection=True, **kwargs):
    if use_rejection:
        return get_surf_pcl_rejection(net, npoints, dim, **kwargs)
    else:
        return get_surf_pcl_langevin_dynamic(net, npoints, dim, **kwargs)


def get_surf_pcl_rejection(
        net, npoints, dim, batch_size=100000, thr=0.05, return_rej_x=False):
    """
    Sampling points with rejection sampling. We first sample uniformly from
    [-1, 1]^3, then reject all points with |distance| > [thr]. Once gathered
    enough rejected points, we will take a gradient step toward the surface:
        y = x - F(x)n(x)
    :param net: Neural field
    :param npoints: Number of points to sample
    :param dim: Dimension of the points
    :param batch_size: Batch size per iteration
    :param thr: Rejection threshold
    :param return_rej_x: Whether returned points right after rejection.
    :return:
        [x] Sampled points
        [rej_x]? Obtained points after rejection.
    """
    out = []
    cnt = 0
    with jittor.no_grad():
        while cnt < npoints:
            x = jittor.rand(1, batch_size, dim).cuda().float() * 2 - 1
            y = jittor.abs(net(x))
            m = (y < thr).view(1, batch_size)
            m_cnt = m.sum().detach().cpu().item()
            if m_cnt < 1:
                continue
            x_eq = x[m].view(m_cnt, dim)
            out.append(x_eq)
            cnt += m_cnt
    rej_x = x = jittor.cat(out, dim=0)[:npoints, :]

    if x.is_leaf:
        x.requires_grad = True
    else:
        x.retain_grad()
    y = net(x)
    g = gradient(y, x).view(npoints, dim).detach()
    g = g / g.norm(dim=-1, keepdim=True)
    x = x - g * y
    if return_rej_x:
        return x, rej_x
    return x


def get_surf_pcl_langevin_dynamic(
        net, npoints, dim, steps=5, eps=1e-4,
        noise_sigma=0.01, filtered=True, sigma_decay=1.,
        max_repeat=10, bound=(1 - 1e-4)):
    out_cnt = 0
    out = None
    already_repeated = 0
    while out_cnt < npoints and already_repeated < max_repeat:
        already_repeated += 1
        x = jittor.rand(npoints, dim).cuda().float() * 2 - 1
        for i in range(steps):
            sigma_i = noise_sigma * sigma_decay ** i
            x = x.detach() + jittor.randn_like(x).to(x) * sigma_i
            x.requires_grad = True
            y = net(x)
            if jittor.allclose(y, jittor.zeros_like(y)):
                break

            g = gradient(y, x).view(npoints, dim).detach()
            g = g / (g.norm(dim=-1, keepdim=True) + eps)
            x = jittor.clamp(x - g * y, min_v=-bound, max_v=bound)

        if filtered:
            with jittor.no_grad():
                y = net(x)
                mask = (jittor.abs(y) < eps).view(-1, 1)
                x = x.view(-1, dim).masked_select(mask).view(-1, dim)
                out_cnt += x.shape[0]
                if out is None:
                    out = x
                else:
                    out = jittor.cat([x, out], dim=0)
        else:
            out = x
            out_cnt = npoints
    out = out[:npoints, :]
    return out


def tangential_projection_matrix(y, x, norm=True, eps=1e-6):
    """
    Compute the tangential projection matrix:
        P = I - n(x)n(x)^T
        where n(x) is the outward surface normal of x
    :param x: (bs, npts, dim) input points
    :param y: (bs, npts, 1) neural_field(x)
    :param norm: Whether normalize the surface normal vector
    :param eps: Numerical eps
    :return:
        [normals] (bs, npts, dim) The surface normal
        [normals_proj] (bs, npts, dim, dim) The projector matrices
    """
    bs, npoints, dim = x.size(0), x.size(1), x.size(2)
    grad = gradient(y, x)
    if norm:
        normals = (
                grad / (grad.norm(dim=-1, keepdim=True) + eps)
        ).view(bs, npoints, dim)
    else:
        normals = grad.view(bs, npoints, dim)
    normals_proj = _addr_(
        jittor.eye(dim).view(1, 1, dim, dim).expand(bs, npoints, -1, -1).to(y),
        normals, normals, alpha=-1
    )
    return normals, normals_proj


def compute_invert_weight(
        x, deform, inp_nf, out_nf, surface=False, normalize=True):
    """
    Computing the weight in Section 5.3.3.

    :param x: (bs, npts, dim) Points from the output space
    :param deform: Network that maps output space to input space
    :param inp_nf: Neural fields of the input space
    :param out_nf: Neural fields of the output space
    :param surface: Whether the inverse is for surface integral.
    :param normalize:
    :return:
    """
    bs, npoints, dim = x.size(0), x.size(1), x.size(2)
    x = x.clone().detach()
    x.requires_grad = True
    y = deform(x).view(bs, npoints, dim)
    J, status = jacobian(y, x)
    assert status == 0

    if surface:
        # Find the change of area along the tangential plane
        yn, yn_proj = tangential_projection_matrix(inp_nf(y), y)
        xn, xn_proj = tangential_projection_matrix(out_nf(x), x)

        J = jittor.bmm(
            J.view(-1, dim, dim),
            xn_proj.view(-1, dim, dim)
        )
        J = _addr_(J.view(bs, npoints, dim, dim),
                   yn.view(bs, npoints, dim),
                   xn.view(bs, npoints, dim))

    weight = jittor.abs(jittor.linalg.det(J.view(bs * npoints, dim, dim)))
    if int(dim) == 3:
        weight = weight ** 2
    weight = 1. / weight.view(bs, npoints)

    if normalize:
        weight = weight / weight.sum(dim=-1, keepdim=True) * npoints

    return weight


def sample_points(
        npoints, dim=3, sample_surf_points=False,
        inp_nf=None, out_nf=None, deform=None,
        invert_sampling=False,
        detach_weight=True, use_rejection=False):
    """
    Sample points from the neural fields: inp_nf, out_nf, and deform.

    :param npoints: Number of points to sample.
    :param dim: Dimension of the points.
    :param sample_surf_points:
    :param inp_nf: Input neural fields. F: (bs, npts, dim) -> (bs, npts, 1)
    :param out_nf: Output neural fields. G: (bs, npts, dim) -> (bs, npts, 1)
    :param deform: Neural fields that deofrm from output space to input space.
                   (bs, npts, dim) -> (bs, npts, dim)
    :param invert_sampling: Whether sample from [inp_nf] then invert the points
                            through the [deform] to become samples of [out_nf]
    :param detach_weight: Whether detach the weights.
    :param use_rejection: Whether use rejection to sample.
    :return:
        [x] (1, npoints, dim) Sampled points on the surface of [out_nf](x) = 0.
        [weights] the weights for inverting the surface intergral.
    """
    if sample_surf_points:
        if invert_sampling:
            assert deform is not None
            assert inp_nf is not None
            y = get_surf_pcl(
                inp_nf, npoints=npoints, dim=dim, use_rejection=use_rejection)
            x = deform.invert(y, iters=30).detach().cuda().float()

            weight = compute_invert_weight(
                x.view(1, -1, dim),
                deform=deform, inp_nf=inp_nf, out_nf=out_nf, surface=True)
        else:
            assert out_nf is not None
            x = get_surf_pcl(
                out_nf, npoints=npoints, dim=dim, use_rejection=use_rejection
            ).detach().cuda().float()
            weight = jittor.ones(1, npoints).cuda().float()
    else:
        x = jittor.rand(1, npoints, dim).cuda().float() * 2 - 1
        weight = jittor.ones(1, npoints).cuda().float()
        if invert_sampling:
            assert deform is not None
            y = x
            x = deform.invert(y, iters=30).detach().cuda().float()
            weight = compute_invert_weight(
                x.view(1, -1, dim),
                deform=deform, inp_nf=inp_nf, out_nf=out_nf, surface=False)

    x = x.view(1, npoints, dim)
    weight = weight.view(1, npoints)
    if detach_weight:
        weight = weight.detach()
    return x, weight
