import jittor
import os
import numpy as np
from glob import glob
import jittor
import jittor.nn as nn
import time
from skimage.measure import marching_cubes
from sklearn.preprocessing import normalize

class Generator(object):
    def __init__(self, model, exp_name, threshold = 0.05, checkpoint = None, cls_threshold=0.2):
        #self.model = model.to(device)
        self.model = model
        self.model.eval()
        #self.device = device
        self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format( exp_name)
        self.load_checkpoint(checkpoint)
        self.threshold = threshold

        # compute binary cls threshold of Bernoulli logits
        self.cls_logits_threshold = np.log(cls_threshold) - np.log(1. - cls_threshold)

        print("cls_logits_threshold: {}".format(self.cls_logits_threshold))



    def generate_mesh(self, data, voxel_resolution=128, EPS=0, chunk_num=128, num_steps=5):

        start = time.time() 
        inputs = data['inputs']

        # add noises
        sigma = 0.02
        inputs += sigma * jittor.randn(inputs.shape)

        for param in self.model.parameters():
            param.requires_grad = False

        bound = 1.
        points = np.meshgrid(
            np.linspace(-bound, bound, voxel_resolution),
            np.linspace(-bound, bound, voxel_resolution),
            np.linspace(-bound, bound, voxel_resolution)
        )
        points = np.stack(points)
        points = np.swapaxes(points, 1, 2)
        points = points.reshape(3, -1).transpose().reshape(
            [voxel_resolution]*3 + [3]).astype(np.float32)
        #print(points.shape)

        #points = torch.from_numpy(
        #            get_raster_points(
        #                voxel_resolution=voxel_resolution)).to(self.device)
        #points = points.reshape([voxel_resolution]*3 + [3])
        points_gpu = jittor.array(points)

        avgpool3d = nn.AvgPool3d(kernel_size=2, stride=1)
        centers = avgpool3d(jittor.array(np.expand_dims(points_gpu.permute(3,0,1,2), axis=0))).permute(0,2,3,4,1)
        centers = centers.reshape([1,-1,3])
        points_gpu = points_gpu.reshape([1,-1,3])
        centers = centers.detach()
        points_gpu = points_gpu.detach()

        # split into chunks
        center_chunks = jittor.chunk(centers, chunks=chunk_num, dim=1)
        points_gpu_chunks = jittor.chunk(points_gpu, chunks=chunk_num, dim=1)

        # encode
        encoding = self.model.encoder(inputs)

        gradient_cpu = np.zeros((1,0,3))
        centers_sdf_cpu = np.zeros((1,0))
        centers_cpu = np.zeros((1,0,3))
        sign_cpu = np.zeros((1,0))

        points_udf_cpu = np.zeros((1,0))

        print('-------begin computing mask--------')

        for i in range(chunk_num):
            point = points_gpu_chunks[i]
            center = center_chunks[i]

            center = center.detach()
            center.requires_grad = True

            with jittor.no_grad():
                udf, p_r_init = self.model.decoder(point, *encoding)
                sign = (p_r_init>0.).float()*2-1

            center_debug = self.model.decoder(center, *encoding)[0]
            center_debug.sync()
            #centers_df_pred = jittor.clamp(center_debug, min_v=None, max_v=self.threshold)
            #np.savetxt("center_log.txt", center_debug.numpy())
            centers_df_pred = center_debug.minimum(self.threshold)
            # decode
            #centers_df_pred = jittor.clamp(
            #    self.model.decoder(center, *encoding)[0], 
            #    max_v=self.threshold)

            #centers_df_pred.sum().backward()
            center_grad = jittor.grad(centers_df_pred.sum(), center, retain_graph=False)

            #gradient_cpu = np.concatenate([gradient_cpu, center.grad.detach().numpy()], axis=1)
            gradient_cpu = np.concatenate([gradient_cpu, center_grad.detach().numpy()], axis=1)
            centers_cpu = np.concatenate([centers_cpu, center.detach().numpy()], axis=1)
            centers_sdf_cpu = np.concatenate([centers_sdf_cpu, centers_df_pred.detach().numpy()], axis=1)
            sign_cpu = np.concatenate([sign_cpu, sign.detach().numpy()], axis=1)
            points_udf_cpu = np.concatenate([points_udf_cpu, udf.detach().numpy()], axis=1)


        gradient = gradient_cpu.reshape([voxel_resolution-1]*3 + [3])
        centers_df_pred = centers_sdf_cpu.reshape([voxel_resolution-1]*3)
        centers = centers_cpu.reshape([voxel_resolution-1]*3 + [3])
        sign_cpu = sign_cpu.reshape([voxel_resolution]*3)
        points_udf_cpu = points_udf_cpu.reshape([voxel_resolution]*3)

        gradient_norm = np.linalg.norm(gradient, axis=-1)
        #print('gradient norm max: {}'.format(np.max(gradient_norm)))
        #print('gradient norm min: {}'.format(np.min(gradient_norm)))
        #print('gradient > 1 cell: {}'.format(gradient_norm[gradient_norm>1.].shape))
        voxel_gradnorm = (gradient_norm>1.).astype(np.int)

        # generate mask
        mask = np.zeros([voxel_resolution]*3)
        voxel = np.zeros([voxel_resolution-1]*3)

        #max_sum = -1

        for i in range(voxel_resolution-1):
            for j in range(voxel_resolution-1):
                for k in range(voxel_resolution-1):

                    if np.abs(centers_df_pred[i,j,k])>=self.threshold:
                        continue

                    # TODO: gradient normalize
                    closest_surfpt = centers[i,j,k] - normalize(gradient[i,j,k].reshape((-1,1)), axis=0).ravel()*centers_df_pred[i,j,k]

                    # check if closest surfpt is inside this cube
                    if (points[i,j,k,0]-EPS < closest_surfpt[0] < points[i+1,j,k,0]+EPS and 
                        points[i,j,k,1]-EPS < closest_surfpt[1] < points[i,j+1,k,1]+EPS and 
                        points[i,j,k,2]-EPS < closest_surfpt[2] < points[i,j,k+1,2]+EPS):

                        '''
                        grid_sign_sum = 0

                        for ii in range(2):
                            for jj in range(2):
                                for kk in range(2):
                                    grid_sign_sum += sign_by_value[i+ii,j+jj,k+kk]
                        '''

                        # get voxel to visualize
                        voxel[i,j,k] = 1
                        #mask[i+1,j+1,k+1] = 1

                        # 8 corners of a cube
                        for ii in range(2):
                            for jj in range(2):
                                for kk in range(2):

                                    mask[i+ii,j+jj,k+kk] = 1
        '''
        # generate mask
        mask_gradnorm = np.zeros([voxel_resolution]*3)
        #voxel = np.zeros([voxel_resolution-1]*3)

        #max_sum = -1

        for i in range(voxel_resolution-1):
            for j in range(voxel_resolution-1):
                for k in range(voxel_resolution-1):

                    if voxel_gradnorm[i,j,k] == 0:

                        # 8 corners of a cube
                        for ii in range(2):
                            for jj in range(2):
                                for kk in range(2):

                                    mask_gradnorm[i+ii,j+jj,k+kk] = 1
        '''


        #mask_cpu = np.array_split(mask.reshape([1,-1,1]), chunk_num, axis=1)

        print('-------begin computing sign and distance--------')

        #print('points: {}'.format(points))

        #point_chunks = torch.chunk(points_gpu, chunks=chunk_num, dim=1)

        points_sdf_cpu = sign_cpu.copy() * 10
        points_sdf_cpu_filtered = np.zeros((1,0))
        #points_cpu = np.zeros((1,0,3))

        sign_by_value_cpu = sign_cpu.copy() * 10
        sign_by_value_cpu_filtered = np.zeros((1,0))

        points_filtered = jittor.array(points[mask==1]).reshape(1,-1,3)

        #print('points_filtered shape: {}'.format(points_filtered.shape))

        point_chunks = jittor.chunk(points_filtered, chunks=chunk_num, dim=1)

        for i in range(len(point_chunks)):

            #mask_chunk = mask_cpu[i]
            #if mask_chunk.sum() == 0:
            #    continue

            point = point_chunks[i]
            point = point.detach()
            point.requires_grad = True

            #print('point shape: {}'.format(point.shape))

            # compute sign by mere value
            #with torch.no_grad():
            udf, sign_by_value = self.model.decoder(point, *encoding)


            '''
            for j in range(num_steps):
                # generate grid gradient of UDF and OCC
                #if j>0:
                #    dis_pred_laststep = dis_pred

                dis_pred = self.model.decoder(point, *encoding)[0]

                #print('sum: {}'.format((dis_pred<dis_pred_laststep).float().sum()))

                #grid_logits_pred = p_r.logits
                #grid_logits_pred = torch.clamp(
                #    grid_logits_pred,
                #    max=self.cls_logits_threshold,
                #    min=-self.cls_logits_threshold
                #)
                #dis_pred = torch.clamp(
                #    dis_pred,
                #    max=self.threshold
                #)

                grad_outputs = torch.ones_like(dis_pred)
                grid_dis_grad = torch.autograd.grad(dis_pred, [point], grad_outputs=grad_outputs, retain_graph=True)[0]
                grid_dis_grad = F.normalize(grid_dis_grad, dim=-1)

                if j == 0:
                    #grid_dis_grad_init = grid_dis_grad.clone().detach()
                    grid_point_init = point.clone().detach()
                    dis_pred_init = dis_pred.clone().detach()

                # convert point to surface point
                #print(point.shape)
                #print(grid_dis_grad.shape)
                #print(dis_pred.shape)

                # if move nearer than last step, then take this update
                #if j>0:
                #    dis_mask = dis_pred>=dis_pred_laststep
                #    dis_pred[dis_mask] = torch.zeros_like(dis_pred)[dis_mask]

                # debug
                #dis_pred = torch.zeros_like(dis_pred)

                point = point - F.normalize(grid_dis_grad, dim=-1) * dis_pred.unsqueeze(-1)
                point = point.detach()
                point.requires_grad = True

            print('near ratio: {}'.format((dis_pred<dis_pred_init).float().sum()/dis_pred.numel()))

            surf_point = point.detach()
            surf_point.requires_grad = True

            p_r = self.model.decoder(surf_point, *encoding)[1]

            grid_logits_pred = p_r

            #p_r.logits.sum().backward(retain_graph=True)
            grad_outputs = torch.ones_like(grid_logits_pred)
            grid_logits_grad = torch.autograd.grad(grid_logits_pred, [surf_point], grad_outputs=grad_outputs)[0]
            grid_logits_grad = F.normalize(grid_logits_grad,dim=-1)
            #self.model.zero_grad()

            #print(point.grad)

            #dis_pred.sum().backward()
            

            # compute sign
            #print('logit: {}'.format(grid_logits_grad))
            #print('dis: {}'.format(grid_dis_grad))
            #print((grid_logits_grad * grid_dis_grad).sum(-1))

            # debug
            #grid_logits_grad = torch.Tensor([0,1,0]).to(self.device)
            #grid_dis_grad_init = F.normalize(grid_point_init - surf_point, dim=-1)
            #grid_dis_grad_init = torch.Tensor([0,-1,0]).to(self.device)
            '''

            grad_outputs = jittor.ones_like(udf)
            #grid_udf_grad = torch.autograd.grad(udf, [point], grad_outputs=grad_outputs, retain_graph=True)[0]
            grid_udf_grad = jittor.grad(udf, point, retain_graph=False)
            grid_udf_grad = jittor.normalize(grid_udf_grad,dim=-1)

            grad_outputs = jittor.ones_like(sign_by_value)
            #grid_sign_grad = torch.autograd.grad(sign_by_value, [point], grad_outputs=grad_outputs)[0]
            grid_sign_grad = jittor.grad(sign_by_value, point, retain_graph=False)
            grid_sign_grad = jittor.normalize(grid_sign_grad,dim=-1)

            sign = ((grid_udf_grad * grid_sign_grad).sum(-1)>0).float()*2-1

            #grid_point_shift = grid_point_init + grid_logits_grad * dis_pred_init.unsqueeze(-1)
            #with torch.no_grad():
            #    grid_dis_shift = self.model.decoder(grid_point_shift, *encoding)[0]
            #sign = (grid_dis_shift>dis_pred_init).float()*2-1
            udf.sync()
            sign.sync()
            points_df_pred = udf * sign
            
            # DEBUG
            #points_df_pred = (cls_pred.argmax(dim=1)*2-1).float()
            #points_df_pred = p_r_init.logits

            #points_cpu = np.concatenate([points_cpu, point.detach().numpy()], axis=1)
            points_sdf_cpu_filtered = np.concatenate([points_sdf_cpu_filtered, points_df_pred.detach().numpy()], axis=1)
            #points_udf_cpu_filtered = np.concatenate([points_udf_cpu_filtered, dis_pred_init.detach().numpy()], axis=1)
            
            sign_by_value_cpu_filtered = np.concatenate([sign_by_value_cpu_filtered, sign_by_value.detach().numpy()], axis=1)

            '''
            # plot vector field
            point_cpu = point[0].detach().numpy()
            grid_logits_grad_cpu = grid_logits_grad[0].detach().numpy()
            grid_dis_grad_cpu = grid_dis_grad[0].detach().numpy()

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            ax.quiver(point_cpu[:,0],point_cpu[:,1],point_cpu[:,2],grid_logits_grad_cpu[:,0],grid_logits_grad_cpu[:,1],grid_logits_grad_cpu[:,2],length=0.05)

            plt.savefig('chunk_{}_logits.png'.format(i))

            plt.close()
            '''

        # CPU compute mask (could be parallelized)
        #points_df_pred = points_sdf_cpu.reshape([voxel_resolution]*3)
        #points = points_cpu.reshape([voxel_resolution]*3 + [3])

        #points_udf_pred = points_udf_cpu.reshape([voxel_resolution]*3)

        #sign_by_value = sign_by_value_cpu.reshape([voxel_resolution]*3)#.astype(np.bool)

        points_sdf_cpu[mask==1] = points_sdf_cpu_filtered.ravel()
        #points_udf_cpu[mask==1] = points_udf_cpu_filtered.ravel()
        sign_by_value_cpu[mask==1] = sign_by_value_cpu_filtered.ravel()


        #pos_pts = points[np.logical_and(points_df_pred>0, points_df_pred<0.1)]
        #neg_pts = points[np.logical_and(points_df_pred<0, points_df_pred>-0.1)]
        #zero_pts = points[points_df_pred==0]

        print('-------begin marching cube-------')
        verts = []
        faces = []
        verts_nomask = []
        faces_nomask = []
        duration = time.time() - start
        
        #verts, faces, norms, vals = marching_cubes(
        #    points_sdf_cpu,
        #    #np.abs(points_sdf_cpu) * ((sign_by_value_cpu>self.cls_logits_threshold)*2-1), 
        #    0, 
        #    mask=mask.astype(bool))

        

        #verts_nomask, faces_nomask, _, _ = marching_cubes(
        #   points_df_pred, 
        #    0)

        verts_nomask, faces_nomask, _, _ = marching_cubes(
            sign_by_value_cpu, 
            0.,
            mask=mask.astype(bool)
        )

        duration = time.time() - start
        

        verts_udf, faces_udf, _, _ = marching_cubes(
            points_udf_cpu,
            6e-3
        )

        '''
        verts_udf, faces_udf, _, _ = marching_cubes(
            points_udf_cpu,
            0.,
            mask=mask_gradnorm.astype(bool)
        )
        '''

        #print(verts_nomask.shape)  # verts_nomask shape (verts, 3)
        centers = np.array((1,1,1))
        spacing = 2./(voxel_resolution-1)
        scale = (points.reshape([-1, 3]).max(0) - points.reshape([-1, 3]).min(0))[0]
        pts_center = points.reshape([-1, 3]).mean(0)


        verts_nomask_normalized = (verts_nomask*spacing - centers)*0.5

        verts_nomask = verts_nomask_normalized*scale + pts_center


        return verts, faces, verts_nomask, faces_nomask, duration, voxel, verts_udf, faces_udf, voxel_gradnorm#, pos_pts, neg_pts, zero_pts



    def generate_point_cloud(self, data, num_steps = 5, num_points = 900000, filter_val = 0.009):

        jittor.gc()
        start = time.time()
        inputs = data['inputs']


        for param in self.model.parameters():
            param.requires_grad = False

        sample_num = 100000
        samples_cpu = np.zeros((0, 3))
        samples = jittor.rand(1, sample_num, 3).float() * 3 - 1.5
        samples.requires_grad = True

        encoding = self.model.encoder(inputs)

        i = 0
        while len(samples_cpu) < num_points:
            print('iteration', i)

            for j in range(num_steps):
                print('refinement', j)
                df_pred = jittor.clamp(self.model.decoder(samples, *encoding)[0], max_v=self.threshold)

                sample_grad = jittor.grad(df_pred.sum(), samples, retain_graph=False)
                sample_grad.sync()
                #df_pred.sum().backward()

                gradient = sample_grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                inputs = inputs.detach()
                samples = samples - jittor.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  # better use Tensor.copy method?
                samples = samples.detach()
                samples.requires_grad = True


            print('finished refinement')

            if not i == 0:
                samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().numpy()))

            samples = samples.detach()
            df_pred = df_pred.detach()
            df_pred.sync()
            samples.sync()
            samples_tmp = samples[df_pred < 0.03]
            samples_tmp.sync()
            samples = samples_tmp.unsqueeze(0)
            indices = jittor.randint(samples.shape[1], shape=(1, sample_num))
            samples = samples[[[0, ] * sample_num], indices]
            samples += (self.threshold / 3) * jittor.randn(samples.shape)  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            print(samples_cpu.shape)

        duration = time.time() - start

        return samples_cpu, duration



    def load_checkpoint(self, checkpoint):
        checkpoints = glob(self.checkpoint_path + '/*')
        #print(checkpoints)
        if checkpoint is None:
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))
                return 0, 0

            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)

            for name in glob(self.checkpoint_path + '/*'):
                if str(checkpoints[-1]) in name:
                    path = self.checkpoint_path + os.path.basename(name)

            #path = self.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
            #    *[*convertSecs(checkpoints[-1]), checkpoints[-1]])
        else:
            path = self.checkpoint_path + '{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path)) 
        '''
        #torch to jittor
        
        import torch
        check_torch = torch.load(path)
        if 'module' in list(check_torch['model_state_dict'].keys())[0]:
            self.model.load_state_dict({k[7:]:v for k,v in check_torch['model_state_dict'].items()})
        else:
            self.model.load_state_dict(check_torch['model_state_dict'])
        epoch = check_torch['epoch']
        training_time = check_torch['training_time']
        del torch
        '''
        checkpoint = jittor.load(path)
        if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
            self.model.load_state_dict({k[7:]:v for k,v in checkpoint['model_state_dict'].items()})
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        
        return epoch, training_time


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds
