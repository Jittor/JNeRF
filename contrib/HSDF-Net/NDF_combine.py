from torch.distributions.utils import probs_to_logits
from trimesh.util import sigfig_round
import models.local_model as model
from models.data import dataloader_garments, voxelized_data_shapenet

from models import generation
import torch
from torch.nn import functional as F
import mesh_to_sdf
import numpy as np

chunk_num = 50
cls_threshold=0.2
cls_logits_threshold = np.log(cls_threshold) - np.log(1. - cls_threshold)
usesign_threshold = 0.005

def rot_YZ(points):
    points_rot = points.copy()
    points_rot[:, 1], points_rot[:, 2] = points[:, 2], points[:, 1]
    return points_rot

def to_grid(points):
    grid_points = points.copy()
    grid_points[:, 0], grid_points[:, 2] = points[:, 2], points[:, 0]

    return 2 * grid_points

def from_grid(grid_points):
    points = grid_points.copy()
    points[:, 0], points[:, 2] = grid_points[:, 2], grid_points[:, 0]

    return 0.5 * points

# 'test', 'val', 'train'
def loadNDF(index, pointcloud_samples, exp_name, data_dir, split_file, sample_distribution, sample_sigmas, res,  mode = 'test'):

    global encoding
    global net
    global device

    net = model.NDF()

    device = torch.device("cuda")

    '''
    if 'garments' in exp_name.lower() :

        dataset = dataloader_garments.VoxelizedDataset(mode =  mode,  data_path = data_dir, split_file = split_file,
                                                        res = res, density =0, pointcloud_samples = pointcloud_samples,
                                                       sample_distribution=sample_distribution,
                                                       sample_sigmas=sample_sigmas,
                                                        )



        checkpoint = 'checkpoint_127h:6m:33s_457593.9149734974'

        generator = generation.Generator(net,exp_name, checkpoint = checkpoint, device = device)

    if 'cars' in exp_name.lower() :

        dataset = voxelized_data_shapenet.VoxelizedDataset( mode = mode, res = res, pointcloud_samples =  pointcloud_samples,
                                                   data_path = data_dir, split_file = split_file,
                                                   sample_distribution = sample_distribution, sample_sigmas = sample_sigmas,
                                                   batch_size = 1, num_sample_points = 1024, num_workers = 1
                                                   )



        checkpoint = 'checkpoint_108h:5m:50s_389150.3971107006'

        generator = generation.Generator(net, exp_name, checkpoint=checkpoint, device=device)
    '''

    dataset = voxelized_data_shapenet.VoxelizedDataset( mode = mode, res = res, pointcloud_samples =  pointcloud_samples,
                                                data_path = data_dir, split_file = split_file,
                                                sample_distribution = sample_distribution, sample_sigmas = sample_sigmas,
                                                batch_size = 1, num_sample_points = 1024, num_workers = 1
                                                )



    generator = generation.Generator(net, exp_name, device=device)

    example = dataset[index]

    print('Object: ',example['path'])
    inputs = torch.from_numpy(example['inputs']).unsqueeze(0).to(device) # lead inputs and samples including one batch channel

    for param in net.parameters():
        param.requires_grad = False

    encoding = net.encoder(inputs)

    return example['path']



def predictRotNDF(points):

    points = rot_YZ(points)
    points = to_grid(points)
    points = torch.from_numpy(points).unsqueeze(0).float().to(device)

    points_chunk = torch.chunk(points, chunk_num, dim=1)

    ndf = np.zeros((0))

    for point in points_chunk:
        point = point.detach()
        point.requires_grad = True

        dist = net.decoder(point,*encoding)[0]
        dist = torch.clamp(dist, max=0.1)
        #logits = p_r.logits

        '''
        mask = dist<usesign_threshold

        if mask.any():
            point_selected = point[mask].detach().unsqueeze(0)
            point_selected.requires_grad = True

            dist_s, p_r_s = net.decoder(point_selected, *encoding)
            logits = p_r_s.logits

            #grad_outputs = torch.ones_like(dist_s)
            #grid_dis_grad = torch.autograd.grad(dist_s, [point_selected], grad_outputs=grad_outputs, retain_graph=True)[0]
            #grid_dis_grad = F.normalize(grid_dis_grad, dim=-1)

            #grad_outputs = torch.ones_like(logits)
            #grid_logit_grad = torch.autograd.grad(logits, [point_selected], grad_outputs=grad_outputs, retain_graph=True)[0]
            #grid_logit_grad = F.normalize(grid_logit_grad, dim=-1)

            sign_selected = (logits>cls_logits_threshold).float()*2-1
            #sign_selected = ((grid_logit_grad * grid_dis_grad).sum(-1)>0).float()*2-1

            dist[mask] = dist_s*sign_selected

            #sign = torch.ones_like(sign_all)

            #sign[dist<usesign_threshold] = sign_all[dist<usesign_threshold]
        '''

        ndf = np.concatenate([ndf, (dist).squeeze(0).detach().cpu().numpy()], axis=0)

    return ndf

def predictGtNDF(points, trimesh_mesh):

    points = rot_YZ(points)
    #points = to_grid(points)

    df, grad = mesh_to_sdf.mesh_to_sdf(trimesh_mesh, points, surface_point_method='sample', return_gradients=True)

    return df

def predictRotGradientNDF(points):
    points = rot_YZ(points)
    points = to_grid(points)
    points = torch.from_numpy(points).unsqueeze(0).float().to(device)
    #points.requires_grad = True

    points_chunk = torch.chunk(points, chunk_num, dim=1)
    gradient = np.zeros((0,3))

    for point in points_chunk:

        point = point.detach()
        point.requires_grad = True

        df_pred = torch.clamp(net.decoder(point,*encoding)[0], max=0.1)
        #p_r_logits = torch.clamp(net.decoder(points,*encoding)[1].logits, max=cls_logits_threshold+0.5, min=cls_logits_threshold-0.5)
        #p_r_logits = net.decoder(point,*encoding)[1].logits

        df_pred.sum().backward()

        gradient = np.concatenate([gradient, F.normalize(point.grad, dim=2)[0].detach().cpu().numpy()], axis=0)

        #gradient = F.normalize(point.grad, dim=2)[0].detach().cpu().numpy()

    #df_pred = df_pred.detach().squeeze(0).cpu().numpy()
    df_pred = None
    return df_pred, rot_YZ( 2 * from_grid(gradient))

def predictGtGradientNDF(points, trimesh_mesh):
    points = rot_YZ(points)
    #points = to_grid(points)

    df, grad = mesh_to_sdf.mesh_to_sdf(trimesh_mesh, points, surface_point_method='sample', return_gradients=True)

    return df, rot_YZ(grad)