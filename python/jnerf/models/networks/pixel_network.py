import jittor as jt
from jittor import nn
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
def PE(x, L, w):
    pe = list()
    pe.append(x)
    for i in range(L):
        for fn in [jt.sin, jt.cos]:
            pe.append(fn(2. ** i * x * w))
    return jt.concat(pe, -1)

class ResMLP(nn.Module):
    def __init__(self, hidden_ch, img_f_ch=None):
        super(ResMLP, self).__init__()
        if img_f_ch is not None:
            self.img_mlp = nn.Sequential(
                nn.Linear(img_f_ch, hidden_ch),
                nn.ReLU(),
            )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU()
        )

    def execute(self, res_input, img_f_input=None):
        if img_f_input is not None:
            res_input = res_input + self.img_mlp(img_f_input)
        return self.mlp(res_input) + res_input


@NETWORKS.register_module()
class PixelNeRF(nn.Module):
    def __init__(self, img_f_ch=512):
        super(PixelNeRF, self).__init__()
        self.cfg = get_cfg()
        # self.pos_encoder = build_from_cfg(self.cfg.encoder.pos_encoder, ENCODERS)
        # self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)

        self.L_pos = 6
        self.L_dir = 0
        self.w = 1.5
        pos_enc_feats = 3 + 3 * 2 * self.L_pos
        dir_enc_feats = 3 + 3 * 2 * self.L_dir
        self.xd_input_ch = pos_enc_feats + dir_enc_feats

        net_width = 512
        # (L(x),d) -> f
        self.mlp = nn.Sequential(
            nn.Linear(self.xd_input_ch, net_width),
            nn.ReLU(),
        )

        # f1
        n_f1 = 3
        f1 = list()
        for i in range(n_f1):
            f1.append(ResMLP(net_width,img_f_ch))
        self.f1 = nn.ModuleList(f1)
        # f2
        n_f2 = 2
        f2 = list()
        for i in range(n_f2):
            f2.append(ResMLP(net_width))
        self.f2 = nn.ModuleList(f2)

        self.final_layer = nn.Linear(net_width, 4)

    def execute(self, img_feature, x, d):
        # x => tensor(N_Rays, N_Samples, 3)
        # d => tensor(N_Rays, 3)
        # img_feature => tensor(N_References, C, N_Rays, N_Samples)
        n_ref, c, n_ray, n_sample = img_feature.shape
        # print(img_feature.shape, x.shape, d.shape)
        img_feature = img_feature.permute(0, 2, 3, 1)
        # x_encode = self.pos_encoder(x)
        # d_encode = self.dir_encoder(d)

        # xd_input = jt.concat([x_encode, d_encode], dim=-1).unsqueeze(dim=0)
        x_encode = PE(x, self.L_pos, self.w)
        d = jt.normalize(d, p=2.0, dim=-1)
        d_encode = PE(d, self.L_dir, self.w)
        # print("encode: ", d_encode.shape, x_encode.shape, x.shape)
        d_encode = d_encode.unsqueeze(dim=1).expand(x_encode[:, :, :3].shape)
        xd_input = jt.concat([x_encode, d_encode], dim=-1).unsqueeze(dim=0)
        # f is the output of MLP(x,d) => tensor(N_References, N_Rays, N_Samples, 512)
        f = self.mlp(xd_input.expand([n_ref, n_ray, n_sample, self.xd_input_ch]))

        for layer in self.f1:
            f = layer(f, img_feature)

        # average the feature
        f = f.mean(dim=0)

        for layer in self.f2:
            f = layer(f)

        outputs = self.final_layer(f)
        sigma = nn.relu(outputs[..., 0])
        c = jt.sigmoid(outputs[..., 1:])
        print(sigma.shape, c.shape)
        return jt.concat([c, sigma.unsqueeze(-1)], -1)
