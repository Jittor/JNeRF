import open3d as o3d
import jittor as jt
import numpy as np
import mcubes
import os
import argparse
from plyfile import PlyData, PlyElement
from jnerf.runner import Runner
from jnerf.utils.config import init_cfg
import time
from tqdm import tqdm
def mesh():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution of space division"
    )
    parser.add_argument(
        "--mcube_smooth",
        type=bool,
        default=False,
        help="use pymcube.smooth function"
    )
    args = parser.parse_args()
    print(args)
    if args.config_file:
        init_cfg(args.config_file)
    runner = Runner()
    runner.load_ckpt(runner.ckpt_path)
    mesh_dir = runner.save_path
    aabb_scale = runner.dataset["train"].aabb_scale
    N = args.resolution
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    zmin, zmax = 0, 1
    xyz_chunk = 512*512*512
    step = max(min(xyz_chunk//(N*N),N),1)
    assert N%step==0
    rgbsigmas=[]
    y = jt.linspace(ymin, ymax, N)
    z = jt.linspace(zmin, zmax, N)
    for k in range(0,N,step):
        rg = (xmax-xmin)/(N-1)
        start = xmin+rg*k
        end = xmin+rg*(k+step-1)
        x = jt.linspace(start, end, step)
        xyz = jt.stack(jt.meshgrid(x, y, z), -1).reshape(-1, 3)
        xyz_ = xyz
        dir_ = jt.zeros_like(xyz_)
        with jt.no_grad():
            B = xyz_.shape[0]
            out_chunks = []
            for i in range(0, B, runner.n_rays_per_batch*128):
                pos=xyz_[i:i + runner.n_rays_per_batch*128]
                dir=dir_[i:i + runner.n_rays_per_batch*128]
                out_chunks += [runner.model(pos, dir)[:,-1]]
            jt.sync_all(True)
            out_chunks = jt.concat(out_chunks,0)
            sigma0 = jt.maximum(out_chunks,0).int()
            rgbsigmas.append(sigma0.numpy())
            jt.sync_all(True)
            jt.gc()
    sigma = np.concatenate(rgbsigmas,0)
    sigma = sigma.reshape(N, N, N)
    if args.mcube_smooth:
        sigma = mcubes.smooth(sigma)
        vertices, triangles = mcubes.marching_cubes(sigma, 0)
    else:
        vertices, triangles = mcubes.marching_cubes(sigma, 0.5)

    vertices_ = (vertices/N).astype(np.float32)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles
    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'),
             PlyElement.describe(face, 'face')]).write(os.path.join(mesh_dir, f'{"mesh-origin"}.ply'))
    print("mesh origin generated mesh-origin.ply")
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, f'{"mesh-origin"}.ply'))
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    face = np.empty(len(mesh.triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = mesh.triangles
    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    # PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
    #         PlyElement.describe(face, 'face')]).write(os.path.join(mesh_dir, f'{"mesh-denoise"}.ply'))
    # print("mesh denoise generated mesh-denoise.ply")
    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    mesh.compute_vertex_normals()
    dir_ = np.asarray(mesh.vertex_normals)

    x_ = vertices_[:, 1].copy()
    y_ = vertices_[:, 0].copy()
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_

    x_ = dir_[:, 1].copy()
    y_ = dir_[:, 0].copy()
    dir_[:, 0] = x_
    dir_[:, 1] = y_

    dir_ = jt.array(dir_)
    vertices_ = jt.array(vertices_)
    rays_o_total = vertices_ - dir_ * 0.2
    rays_o_total = (rays_o_total-0.5)*aabb_scale+0.5
    W, H = runner.image_resolutions
    W = int(W)
    H = int(H)
    fake_img_ids = jt.zeros([H*W], 'int32')
    N_vertices = len(vertices_)
    img = []
    alpha = []
    for start in tqdm(range(0, N_vertices, runner.n_rays_per_batch)):
        with jt.no_grad():
            end = start + runner.n_rays_per_batch
            rays_o = rays_o_total[start:end]
            rays_d = dir_[start:end]
            pos, dir = runner.sampler.sample(fake_img_ids, rays_o, rays_d)
            network_outputs = runner.model(pos, dir)
            rgb, a = runner.sampler.rays2rgb(network_outputs, inference=True)
            img += [rgb.numpy()]
            alpha += [a.numpy()]
            jt.gc()
    img = np.concatenate(img, 0)
    alpha = np.concatenate(alpha, 0)
    img = img + np.array(runner.background_color)*(1 - alpha)
    img = (img*255+0.5).clip(0, 255).astype('uint8')
    img.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_ = np.asarray(vertices_)
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+img.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in img.dtype.names:
        vertex_all[prop] = img[prop][:, 0]
    face = np.empty(len(mesh.triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = mesh.triangles
    PlyData([PlyElement.describe(vertex_all, 'vertex'),
             PlyElement.describe(face, 'face')]).write(os.path.join(mesh_dir, f'{"mesh-color"}.ply'))
    print("mesh color generated mesh-color.ply")


if __name__ == "__main__":
    mesh()