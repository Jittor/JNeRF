import jittor
import jittor.nn as nn

class HSDF(nn.Module):
    
    def __init__(self, hidden_dim=256):
        super(HSDF, self).__init__()
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        
        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)

        # classification head
        self.fc_0_cls = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1_cls = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2_cls = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out_cls = nn.Conv1d(hidden_dim, 1, 1)

        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)

        # add remez_net
        #self.m = 7
        #self.n = 7
        #self.max_mn = self.m if self.m >= self.n else self.n
        #self.rat = rational_net(self.max_mn, self.max_mn, feature_size=feature_size)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = jittor.array(displacments)
        
    def encoder(self,x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = jittor.concat([p + d for d in self.displacments], dim=2)

        
        # feature extraction
        feature_0 = nn.grid_sample(f_0, p, padding_mode='border', align_corners=True)
        feature_1 = nn.grid_sample(f_1, p, padding_mode='border', align_corners=True)
        feature_2 = nn.grid_sample(f_2, p, padding_mode='border', align_corners=True)
        feature_3 = nn.grid_sample(f_3, p, padding_mode='border', align_corners=True)
        feature_4 = nn.grid_sample(f_4, p, padding_mode='border', align_corners=True)
        feature_5 = nn.grid_sample(f_5, p, padding_mode='border', align_corners=True)
        feature_6 = nn.grid_sample(f_6, p, padding_mode='border', align_corners=True)
    

        # here every channel corresponds to one feature.

        features = jittor.concat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = jittor.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = jittor.concat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        p_r = None
        
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        #net = self.fc_out(net)
        out = net.squeeze(1)

        # classification head
        net_cls = self.actvn(self.fc_0_cls(features))
        net_cls = self.actvn(self.fc_1_cls(net_cls))
        net_cls = self.actvn(self.fc_2_cls(net_cls))

        # classification task with no actvn
        out_cls = self.fc_out_cls(net_cls).squeeze(1)  # (B, 1, samples_num) -> (B, samples_num)

        # return occupancy probabilities for the sampled points
        #print(out_cls)
        #p_r = dist.Bernoulli(logits=out_cls)
        

        return  out, out_cls

    def execute(self, p, x):
        out, p_r = self.decoder(p, *self.encoder(x))
        return out, p_r