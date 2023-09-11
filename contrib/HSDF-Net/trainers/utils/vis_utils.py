from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tqdm
#import torch
import jittor
import trimesh
import skimage
import numpy as np
import skimage.measure


def imf2mesh(imf, res=256, threshold=0.0, batch_size = 10000, verbose=True,
             use_double=False, normalize=False, norm_type='res',
             return_stats=False, bound=1.):
    xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
    grid = np.concatenate([
        ys[..., np.newaxis],
        xs[..., np.newaxis],
        zs[..., np.newaxis]
    ], axis=-1).astype(np.float)
    grid = (grid / float(res - 1) - 0.5) * 2 * bound
    grid = grid.reshape(-1, 3)
    # Grid will be [-1, 1] * bound

    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)
    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        with jittor.no_grad():
            xyz = jittor.Var(
                grid[sidx:eidx, :]).float().cuda().view(1, -1, 3)
            if use_double:
                xyz = xyz.double()
            distances = imf(xyz)
            distances = distances.cpu().numpy()
        dists_lst.append(distances.reshape(-1))

    dists = np.concatenate(
        [x.reshape(-1, 1) for x in dists_lst], axis=0).reshape(-1)
    field = dists.reshape(res, res, res)
    try:
        vert, face, _, _ = skimage.measure.marching_cubes(
            field, level=threshold)
        print(vert.max(), vert.min())
        # Vertices will be [0, res - 1]

        if normalize:
            if norm_type == 'norm':
                center = vert.mean(axis=0).view(1, -1)
                vert_c = vert - center
                length = np.linalg.norm(vert_c, axis=-1).max()
                vert = vert_c / length
            elif norm_type == 'res':
                vert = (vert / float(res - 1) - 0.5) * 2 * bound
            else:
                raise ValueError
        new_mesh = trimesh.Trimesh(vertices=vert, faces=face)
    except ValueError as e:
        print(field.max(), field.min())
        print(e)
        new_mesh = None
    except RuntimeError as e:
        print(field.max(), field.min())
        print(e)
        new_mesh = None

    if return_stats:
        if new_mesh is not None:
            area = new_mesh.area
            vol = (field < threshold).astype(np.float).mean() * (2 * bound) ** 3
        else:
            area = 0
            vol = 0
        return new_mesh, {
            'vol': vol,
            'area': area
        }

    return new_mesh


def make_2d_grid(r, add_noise=False):
    xs, ys = jittor.meshgrid(jittor.arange(r), jittor.arange(r))    
    xy = jittor.cat([ys.reshape(-1, 1), xs.reshape(-1, 1)], dim=-1).float()
    if add_noise:
        xy += jittor.rand_like(xy)
    else:
        xy += 0.5
    xy = (xy / float(r) - 0.5) * 2
    return xy


def imf2img(imf, res=256, add_noise=False, batch_size=10000, threshold=0.,
            verbose=False, grid=None, return_stats=False, bound=1):
    if grid is None:
        grid = make_2d_grid(res, add_noise=add_noise).view(-1, 2)
    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)
    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        with jittor.no_grad():
            xyz = grid[sidx:eidx, :].cuda().view(1, -1, 2)
            n = xyz.size(1)
            distances = imf(xyz)
            distances = distances.cpu().numpy()
        dists_lst.append(distances.reshape(n, -1))
    dists = np.concatenate(
        [x for x in dists_lst], axis=0)
    img = dists.reshape(res, res, -1)
    if return_stats:
        area = (img < threshold).astype(np.float).mean() * 2 ** 2
        contours = skimage.measure.find_contours(
            img.reshape(res, res), level=threshold)
        total_length = 0
        for vert in contours:
            n_v_c = vert.shape[0]
            n_v_c_idx = np.array(
                (np.arange(n_v_c).astype(np.int) + 1) % n_v_c).astype(np.int)
            v_next = vert[n_v_c_idx, :]
            v_next = v_next.reshape(n_v_c, 2)
            diff = (vert - v_next) / float(res)
            dist = np.linalg.norm(diff, axis=-1).sum()
            total_length += dist
        return img, {
            'area' : area,
            'len': total_length,
            'contours': contours
        }
    return img
