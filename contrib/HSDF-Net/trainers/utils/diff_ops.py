# Based on https://github.com/vsitzmann/siren/blob/master/diff_operators.py
#import torch
#from torch.autograd import grad
import jittor


def hessian(y, x):
    """
    Hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations, dim, channels)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = jittor.ones_like(y[..., 0]).to(y.device)
    h = jittor.zeros(meta_batch_size, num_observations,
                    y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y,
                                   create_graph=True)[0][..., :]

    status = 0
    if jittor.any(jittor.isnan(h)):
        status = -1
    return h, status


def laplace(y, x, normalize=False, eps=0., return_grad=False):
    grad = gradient(y, x)
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)
    div = divergence(grad, x)

    if return_grad:
        return div, grad
    return div


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(
            y[..., i], x, jittor.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = jittor.ones_like(y)
    grad = jittor.grad(
        y, [x], retain_graph=True)[0]

    #print('grad: {}'.format(grad))
    return grad


def jacobian(y, x):
    """
    Jacobian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    ret: shape (meta_batch_size, num_observations, channels, dim)
    """
    meta_batch_size, num_observations = y.shape[:2]
    # (meta_batch_size*num_points, 2, 2)
    jac = jittor.zeros(
        meta_batch_size, num_observations,
        y.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(
            y_flat, x, jittor.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if jittor.any(jittor.isnan(jac)):
        status = -1

    return jac, status

