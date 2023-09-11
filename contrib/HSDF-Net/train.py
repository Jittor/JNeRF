import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import jittor
import configs.config_loader as cfg_loader
import os

jittor.flags.use_cuda=1

cfg = cfg_loader.get_config()

# # configure apex
# torch.cuda.set_device(cfg.local_rank)
# torch.distributed.init_process_group(
#     'nccl',
#     init_method='env://'
# )

net = model.HSDF()

print("local rank: {}".format(cfg.local_rank))

train_dataset = voxelized_data.VoxelizedDataset('train',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=2,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)
val_dataset = voxelized_data.VoxelizedDataset('val',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=2,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)

# # debug for NaN
# torch.autograd.set_detect_anomaly(True)

trainer = training.Trainer(net,
                           #torch.device('cuda:{}'.format(cfg.local_rank)),
                           train_dataset,
                           val_dataset,
                           cfg.exp_name,
                           optimizer=cfg.optimizer,
                           lr=cfg.lr,
                        #    local_rank=cfg.local_rank,
                           cls_threshold=cfg.threshold)

trainer.train_model(cfg.num_epochs)
