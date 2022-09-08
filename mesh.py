import open3d as o3d
import jittor as jt
import numpy as np
import mcubes
import trimesh
from plyfile import PlyData, PlyElement
from jnerf.runner import Runner 
from jnerf.utils.config import init_cfg


def mesh():
    init_cfg('./projects/ngp/configs/ngp_base.py')
    runner = Runner()
    runner.load_ckpt(runner.ckpt_path)
    N = 128
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    zmin, zmax = 0, 1
    x = jt.linspace(xmin, xmax, N)
    y = jt.linspace(ymin, ymax, N)
    z = jt.linspace(zmin, zmax, N)
    xyz_ = jt.stack(jt.meshgrid(x, y, z), -1).reshape(-1, 3)
    dir_ = jt.zeros_like(xyz_)
    with jt.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, runner.n_rays_per_batch):
            out_chunks += [runner.model(xyz_[i:i + runner.n_rays_per_batch], dir_[i:i + runner.n_rays_per_batch])]
        rgbsigma = np.concatenate(out_chunks, 0)
    sigma = rgbsigma[:, -1]
    sigma = np.maximum(sigma, 0)
    sigma = sigma.reshape(N, N, N)
    sigma = sigma.astype(int)
    vertices, triangles = mcubes.marching_cubes(sigma, 0.5)
    mesh_o = trimesh.Trimesh(vertices / N, triangles)
    vertices_ = (vertices/N).astype(np.float32)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [('x', 'f4'), ('y','f4'), ('z', 'f4')]
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles
    print("mesh 已经生成 mesh.ply")
    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
            PlyElement.describe(face, 'face')]).write(f'{"mesh"}.ply')
    mesh = o3d.io.read_triangle_mesh(f"{'mesh'}.ply")
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    face = np.empty(len(mesh.triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = mesh.triangles
    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
            PlyElement.describe(face, 'face')]).write(f'{"mesh-noise"}.ply')
    print("mesh 去噪 已经生成 mesh-noise.ply")

if __name__ == "__main__":
    mesh()