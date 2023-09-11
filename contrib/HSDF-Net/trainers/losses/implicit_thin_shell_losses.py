#import torch
#import torch.nn.functional as F
import jittor
import jittor.nn as F
from trainers.utils.diff_ops import hessian, jacobian
from trainers.utils.igp_utils import sample_points, tangential_projection_matrix


def bending_loss(
        inp_nf, out_nf,
        # Presampled points
        x=None, weights=None,
        # Sampling
        npoints=1000, dim=3, use_surf_points=False, deform=None,
        invert_sampling=False, detach_weight=True, use_rejection=False,
        # Loss related
        loss_type='l2', reduction='mean',
):
    if x is None:
        x, weights = sample_points(
            npoints, dim=dim, sample_surf_points=use_surf_points,
            inp_nf=inp_nf, out_nf=out_nf, deform=deform,
            invert_sampling=invert_sampling,
            detach_weight=detach_weight, use_rejection=use_rejection,
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        assert weights is not None
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    # Compute Hessian from the output space
    if x.is_leaf:
        x.requires_grad = True
    else:
        x.retain_grad()
    y_out = out_nf(x)
    h_out, h_out_status = hessian(y_out, x)
    h_out = h_out.view(bs * npoints, dim, dim)

    # Compute the projection matrix from the output space
    _, P = tangential_projection_matrix(y_out, x)
    P = P.view(bs * npoints, dim, dim)

    # Compute points from the input space
    x_inp = deform(x).view(bs, npoints, dim)
    J, J_status = jacobian(x_inp, x)
    J = J.view(bs * npoints, dim, dim)

    # Compute Hessian from the input space
    x_inp.retain_grad()
    y_inp = inp_nf(x_inp)
    h_inp, h_inp_status = hessian(y_inp, x_inp)
    h_inp = h_inp.view(bs * npoints, dim, dim)

    # Compute the projected hessians and their differences after adjustment
    h_inp_J = jittor.bmm(J.transpose(1, 2).contiguous(), jittor.bmm(h_inp, J))
    diff = jittor.bmm(
        P.transpose(1, 2).contiguous(), jittor.bmm(h_out - h_inp_J, P))

    # Compute the Forbinius norm (weighted)
    F_norm = diff.view(bs * npoints, -1).norm(dim=-1, keepdim=False)
    F_norm = F_norm.view(bs, npoints)
    F_norm = F_norm * weights

    if loss_type == 'l2':
        loss = F.mse_loss(
            F_norm, jittor.zeros_like(F_norm), reduction=reduction)
    elif loss_type == 'l1':
        loss = F.l1_loss(
            F_norm, jittor.zeros_like(F_norm), reduction=reduction)
    else:
        raise ValueError
    return loss


def stretch_loss(
        inp_nf, out_nf, deform,
        x=None, npoints=1000, dim=3, use_surf_points=False, invert_sampling=False,
        loss_type='l2', reduction='mean', weights=1,
        detach_weight=True, use_rejection=False,
):
    if x is None:
        x, weights = sample_points(
            npoints, dim=dim, sample_surf_points=use_surf_points,
            inp_nf=inp_nf, out_nf=out_nf, deform=deform,
            invert_sampling=invert_sampling,
            detach_weight=detach_weight, use_rejection=use_rejection,
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        assert weights is not None
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    # Compute Projection on the output space
    if x.is_leaf:
        x.requires_grad = True
    x.retain_grad()
    y_out = out_nf(x)
    _, P = tangential_projection_matrix(y_out, x)
    P = P.view(bs * npoints, dim, dim)

    # Compute the deformation Jacobian
    x_inp = deform(x).view(bs, npoints, dim)
    J, J_status = jacobian(x_inp, x)
    J = J.view(bs * npoints, dim, dim)

    # Compute the matrix of interests
    I = jittor.eye(dim).view(1, dim, dim).to(J)
    diff = I - jittor.bmm(J.transpose(1, 2), J)
    diff = jittor.bmm(P.transpose(1, 2), jittor.bmm(diff, P))

    # Compute the Forbinius norm (weighted)
    F_norm = diff.view(bs * npoints, -1).norm(dim=-1, keepdim=False)
    F_norm = F_norm.view(bs, npoints)
    F_norm = F_norm * weights

    if loss_type == 'l2':
        loss = F.mse_loss(
            F_norm, jittor.zeros_like(F_norm), reduction=reduction)
    elif loss_type == 'l1':
        loss = F.l1_loss(
            F_norm, jittor.zeros_like(F_norm), reduction=reduction)
    else:
        raise ValueError
    return loss

