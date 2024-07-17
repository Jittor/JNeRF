import trimesh
import igl
import numpy as np
import glob
import multiprocessing as mp
from multiprocessing import Pool
import os
import traceback
from functools import partial
import random
import gc
import configs.config_loader as cfg_loader
from mesh_to_sdf import mesh_to_sdf

# number of distance field samples generated per object
sample_num = 1000000

def boundary_sampling(path, sigma):
    try:

        out_path = os.path.dirname(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path,file_name + '_scaled.off')
        out_file = out_path + '/boundary_{}_samples.npz'.format( sigma)

        if os.path.exists(out_file):
            print('Exists: {}'.format(out_file))
            return

        print('processing {}'.format(input_file))

        mesh = trimesh.load(input_file)
        points = mesh.sample(sample_num)

        if sigma == 0:
            boundary_points = points
        else:
            boundary_points = points + sigma * np.random.randn(sample_num, 3)

        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        #print('before mesh_to_sdf')

        if sigma == 0:
            df = np.zeros(boundary_points.shape[0])
        else:
            #df = np.abs(igl.signed_distance(boundary_points, mesh.vertices, mesh.faces)[0])
            df = igl.signed_distance(boundary_points, mesh.vertices, mesh.faces, True)[0]
            # df = mesh_to_sdf(mesh, boundary_points, sign_method='normal', surface_point_method='sample')

        #print('after mesh_to_sdf')

        np.savez(out_file, points=boundary_points, df = df, grid_coords= grid_coords)
        print('Finished: {}'.format(path))

    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


    del mesh, df, boundary_points, grid_coords, points
    gc.collect()
