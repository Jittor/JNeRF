from __future__ import division
import jittor
import jittor.optim as optim
import jittor.nn as nn

from trainers.losses.filtering_losses import loss_lap, loss_lap_dsdf
from trainers.losses.eikonal_loss import loss_eikonal, loss_eikonal_dsdf

from glob import glob
import numpy as np
import time
import os

from tensorboardX import SummaryWriter

class Trainer(object):

    def __init__(
        self, 
        model, 
        #device, 
        train_dataset, 
        val_dataset, 
        exp_name, 
        optimizer='Adam', 
        lr = 1e-4, 
        threshold = 0.1, 
        cls_threshold=0.2):

        self.model = model

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr= lr)
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)


        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( exp_name)
        if jittor.rank == 0:
            if not os.path.exists(self.checkpoint_path):
                print(self.checkpoint_path)
                os.makedirs(self.checkpoint_path)
            self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.max_dist = threshold

        # compute binary cls threshold of Bernoulli logits
        self.cls_logits_threshold = np.log(cls_threshold) - np.log(1. - cls_threshold)


    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, acc = self.compute_loss(batch)
        #loss.backward()
        #self.optimizer.step(loss)
        print("loss {}".format(loss))
        self.optimizer.backward(loss)
        self.optimizer.step()

        # map-reduce loss and acc
        reduced_loss, reduced_acc = loss, acc
        if jittor.in_mpi:
            reduced_loss = loss.mpi_all_reduce()# / jittor.world_size
            reduced_acc = acc.mpi_all_reduce()# / jittor.world_size

        return reduced_loss.item(), reduced_acc.item()

    def compute_loss(self,batch):
        #device = self.device

        p = batch.get('grid_coords')#.to(device)

        # clamp p to -1,1
        #p = torch.clamp(p, max=1., min=-1.)

        # for computing curvature
        #p.requires_grad = True

        df_gt = batch.get('df')#.to(device) #(Batch,num_points)
        inputs = batch.get('inputs')#.to(device)

        #print('input max: {}'.format(torch.max(inputs)))
        #print('input min: {}'.format(torch.min(inputs)))

        #print('p max: {}'.format(torch.max(p)))
        #print('p min: {}'.format(torch.min(p)))


        df_pred, p_r = self.model(p,inputs) #(Batch,num_points)

        #print('df_pred max: {}'.format(torch.max(df_pred)))
        #print('df_pred min: {}'.format(torch.min(df_pred)))

        # have to split abs val traning and sign traning, cauz they're conflicting at open regions
        # regression loss
        loss_r = nn.L1Loss()(
            jittor.clamp(df_pred, max_v=self.max_dist, min_v=0.),
            jittor.clamp(jittor.abs(df_gt), max_v=self.max_dist))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        
        #loss_r = torch.nn.L1Loss(reduction='none')(
        #    df_pred,
        #    torch.abs(df_gt))# out = (B,num_points) by componentwise comparing vecots of size num_samples:

        loss_c = nn.L1Loss()(
            jittor.clamp(p_r, max_v=self.max_dist, min_v=-self.max_dist),
            jittor.clamp(df_gt, max_v=self.max_dist, min_v=-self.max_dist))# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        

        # classification loss
        #sign_gt = torch.sign(df_gt)

        #cls_gt = (sign_gt+1)/2

        #loss_c = torch.nn.CrossEntropyLoss()(
        #    cls_pred.permute(0,2,1).reshape(-1,2), 
        #   cls_gt.view(-1))

        # soft label
        #cls_gt = cls_gt - sign_gt * 0.5 * torch.exp(-1/torch.abs(df_gt))

        #loss_c = F.binary_cross_entropy_with_logits(
        #    p_r.logits, cls_gt, reduction='none'#, weight = 1/torch.exp(df_pred.detach())
        #)

        # introduce curvature loss for sharpening
        '''
        loss_lap_scaling = 1. * loss_lap_dsdf(
            df_pred,
            df_pred,
            x = p,
            npoints = 5000,
            beta = 0.,
            masking_thr = 50
        )

        loss_unit_grad_norm = loss_eikonal_dsdf(
            y = df_pred,
            x = p,
            weights = 50.
        )
        '''

        #print('loss_lap_scaling: {}'.format(loss_lap_scaling.sum(-1).mean()))
        #print('loss_unit_grad_norm: {}'.format(loss_unit_grad_norm.sum(-1).mean()))
        

        # compute accuracy for multi-class classification
        
        acc = ((p_r>0.).long()==(df_gt>0.).long()).sum().double() / (df_gt.shape[0] * df_gt.shape[1])
        #acc = torch.zeros((1))

        #print('loss_r: {}'.format(loss_r.sum(-1).mean()))
        #print('loss_c: {}'.format(loss_c.sum(-1).mean()))

        loss = loss_r.sum(-1).mean() + loss_c.sum(-1).mean()# + loss_unit_grad_norm.sum(-1).mean() + loss_lap_scaling.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

        return loss, acc

    def train_model(self, epochs):
        loss = 0
        train_data_loader = self.train_dataset
        start, training_time = self.load_checkpoint()
        iteration_start_time = time.time()

        for epoch in range(start, epochs):
            sum_loss = 0
            sum_acc = 0
            if jittor.rank == 0:
                print('Start epoch {}'.format(epoch))

            # shuffle ditributed sampler
            #train_data_loader.sampler.set_epoch(epoch)

            print("loader length: {}".format(len(train_data_loader)))

            for idx, batch in enumerate(train_data_loader):
                #print('idx: {}'.format(idx))

                if jittor.rank == 0:
                    #save model
                    iteration_duration = time.time() - iteration_start_time
                    if iteration_duration > 60 * 60:  # eve model every X min and at start
                        #print('{} eval'.format(self.local_rank))
                        
                        training_time += iteration_duration
                        iteration_start_time = time.time()

                        self.save_checkpoint(epoch, training_time)
                        val_loss, val_acc = self.compute_val_loss()

                        if self.val_min is None:
                            self.val_min = val_loss

                        if val_loss < self.val_min:
                            self.val_min = val_loss
                            for path in glob(self.exp_path + 'val_min=*'):
                                os.remove(path)
                            np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])

                        self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                        self.writer.add_scalar('val acc batch avg', val_acc, epoch)


                #optimize model
                loss, acc = self.train_step(batch)
                if jittor.rank == 0:
                    print("Current loss: {} acc: {}".format(loss / self.train_dataset.num_sample_points, acc))
                sum_loss += loss
                sum_acc += acc



            if jittor.rank == 0:
                self.writer.add_scalar('training loss last batch', loss, epoch)
                self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)
                self.writer.add_scalar('training acc batch avg', sum_acc / len(train_data_loader), epoch)




    def save_checkpoint(self, epoch, training_time):
        path = self.checkpoint_path + 'checkpoint_{}h_{}m_{}s_{}.tar'.format(*[*convertSecs(training_time),training_time])
        if not os.path.exists(path) and jittor.rank == 0:
            jittor.save({ #'state': torch.cuda.get_rng_state_all(),
                        'training_time': training_time ,'epoch':epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)



    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0,0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)

        for name in glob(self.checkpoint_path + '/*'):
            if str(checkpoints[-1]) in name:
                path = self.checkpoint_path + os.path.basename(name)
        #path = self.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(checkpoints[-1]),checkpoints[-1]])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = jittor.load(path)

        # load partially
        #print(checkpoint['optimizer_state_dict']['param_groups'])
        #exit()
        #print(self.optimizer.state_dict()['param_groups'])
        #exit()

        if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('load from without apex')
            self.model.load_state_dict({'module.'+k:v for k,v in checkpoint['model_state_dict'].items()})

        epoch = 0
        training_time = 0

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #amp.load_state_dict(checkpoint['amp_state_dict'])
            epoch = checkpoint['epoch']
            training_time = checkpoint['training_time']
        except:
            print('find pretrained weights, epoch reduce to 0')
        # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.
        return epoch, training_time

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        sum_val_acc = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.__iter__()
                val_batch = self.val_data_iterator.__next__()

            #val_loss, val_acc = self.compute_loss( val_batch)
            sum_val_loss += self.compute_loss( val_batch)[0].data.item()
            sum_val_acc += self.compute_loss( val_batch)[1].data.item()

        return sum_val_loss / num_batches, sum_val_acc / num_batches 

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
