import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models.generation import Generator
import jittor
import configs.config_loader as cfg_loader
import os
import trimesh
import numpy as np
from tqdm import tqdm
from utils import voxel2obj

jittor.flags.use_cuda=1

cfg = cfg_loader.get_config()

#device = torch.device("cuda")
net = model.HSDF()

dataset = voxelized_data.VoxelizedDataset('test',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=1,
                                          num_sample_points=cfg.num_sample_points_generation,
                                          num_workers=8,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)

gen = Generator(net, cfg.exp_name, cls_threshold=cfg.threshold)

out_path = 'experiments/{}/evaluation_0.02_128/'.format(cfg.exp_name)


def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    #loader = dataset.get_loader(shuffle=True)

    for i, data in tqdm(enumerate(dataset)):

        path = os.path.normpath(data['path'][0])

        '''
        # selected testing cases for car
        if 'bceb15ddfb9fe56aa13d6c605d0084d3' not in path \
            and '2b4664cf53176418faeea7738551d104' not in path \
            and 'a7c0f3bcc2347710a312b3f0b49ff828' not in path \
            and '2e8c4fd40a1be2fa5f38ed4497f2c53c' not in path \
            and '32924c86ee4a2c69aa4eefa8f42b566e' not in path \
            and '7e412497b8ab74963f2c3a55558a78f' not in path \
            and '7d33cb52d556c1fe618e9d35559b7aa' not in path \
            and '5ce5d2b8c3a7b846d13b7e043606607d' not in path \
            and '1f43243b81df21277925d1ea63246010' not in path \
            and '1ffe99aba88ca413ca71c17c1eef7213' not in path:
            continue
        '''

        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            continue
        else:
            os.makedirs(export_path)

        for num_steps in [7]:
            
            #debug
            verts, faces, verts_nomask, faces_nomask, duration, voxel, verts_udf, faces_udf, voxel_gradnorm = gen.generate_mesh(data, voxel_resolution=128, chunk_num=16)
            #verts, faces, verts_nomask, faces_nomask, duration, voxel, verts_udf, faces_udf, voxel_gradnorm = gen.generate_mesh(data, voxel_resolution=512, chunk_num=4096)
            #np.savez(export_path + 'dense_point_cloud_{}'.format(num_steps), point_cloud=verts, duration=duration)
            #np.savez(export_path + 'dense_point_cloud_{}_nomask'.format(num_steps), point_cloud=verts_nomask, duration=duration)
            print('num_steps', num_steps, 'duration', duration)
            #trimesh.Trimesh(vertices=verts, faces=faces).export(
            #    export_path + 'dense_point_cloud_{}.off'.format(num_steps))
            trimesh.Trimesh(vertices=verts_nomask, faces=faces_nomask).export(
                export_path + 'dense_point_cloud_{}_nomask.off'.format(num_steps))
            trimesh.Trimesh(vertices=verts_udf, faces=faces_udf).export(
                export_path + 'dense_point_cloud_{}_udf.off'.format(num_steps))

            #trimesh.Trimesh(vertices=pos_pts, faces=[]).export(
            #    export_path + 'dense_point_cloud_{}_pos.off'.format(num_steps))
            #trimesh.Trimesh(vertices=neg_pts, faces=[]).export(
            #    export_path + 'dense_point_cloud_{}_neg.off'.format(num_steps))
            #trimesh.Trimesh(vertices=zero_pts, faces=[]).export(
            #    export_path + 'dense_point_cloud_{}_zero.off'.format(num_steps))

            voxel2obj(export_path + 'voxel_gradnorm.obj', voxel_gradnorm)
            

            pc, duration = gen.generate_point_cloud(data, num_steps)
            print('pc duration', duration)
            trimesh.Trimesh(vertices=pc, faces=[]).export(
                export_path + 'dense_point_cloud_{}_pc.off'.format(num_steps))



gen_iterator(out_path, dataset, gen)
