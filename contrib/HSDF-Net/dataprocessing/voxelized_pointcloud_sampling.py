from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
import os
import traceback

kdtree, grid_points, cfg = None, None, None
def voxelized_pointcloud_sampling(path):
    try:

        out_path = os.path.dirname(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path,file_name + '_scaled.off')
        out_file = out_path + '/voxelized_point_cloud_{}res_{}points.npz'.format(cfg.input_res, cfg.num_points)


        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            return



        mesh = trimesh.load(input_file)
        point_cloud = mesh.sample(cfg.num_points)

        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)

        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = cfg.bb_min, bb_max = cfg.bb_max, res = cfg.input_res)
        print('Finished: {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

def init(cfg_param):
    global kdtree, grid_points, cfg
    cfg = cfg_param
    grid_points = create_grid_points_from_bounds(cfg.bb_min, cfg.bb_max, cfg.input_res)
    kdtree = KDTree(grid_points)

def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list