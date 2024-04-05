import numpy as np
import jittor as jt
from jittor import nn
import numpy as np


# Misc
img2mse = lambda x, y : jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.array(np.array([10.])))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [jt.sin, jt.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

#tree class
class Node():
    def __init__(self, anchors, sons, linears):
        self.anchors = anchors
        self.sons = sons
        self.linears = linears

# Model
class OutputNet(nn.Module):
    def __init__(self, W, input_ch_views):
        """ 
        """
        super(OutputNet, self).__init__()

        self.rgb_linear = nn.Linear(W, 3)
        
    def execute(self, h, input_views):

        rgb = self.rgb_linear(h)
        return rgb

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, head_num=8, threshold=3e-2):
        """ 
        """
        super(NeRF, self).__init__()
        D=12
        print("input_ch",input_ch,"input_ch_views",input_ch_views,"output_ch",output_ch,"skips",skips,"use_viewdirs",use_viewdirs)
        # W=128
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        skips=[]
        # skips = [2,4,6]
        self.skips = skips
        # self.ress = [1,3,7,11]
        # self.ress = []
        # self.outs = [1,3,7,11]
        self.force_out = [0]
        # self.force_out = [7,8,9,10,11,12,13,14]
        self.use_viewdirs = use_viewdirs
        assert self.use_viewdirs==False

        self.threshold = threshold
        
        self.build_tree(head_num)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skip_linear else nn.Linear(W + input_ch, W) for i in range(self.linear_num-1)])
        # for i in range(self.nlinear_list[0]+1,len(self.pts_linears)):
        #     jt.init.constant_(self.pts_linears[i].weight, 0.0)
        #     jt.init.constant_(self.pts_linears[i].bias, 0.0)
        # self.confidence_linears = nn.ModuleList([nn.Linear(W+ input_ch, 1) for i in range(D)])
        self.confidence_linears = nn.ModuleList([nn.Linear(W, 1) for i in range(self.node_num)])
        # self.outnet = OutputNet(W, input_ch_views)
        self.outnet = nn.ModuleList([OutputNet(W, input_ch_views) for i in range(self.node_num)])
    
    def get_anchor(self, i):
        return getattr(self, i)

    def get_son_list(self, son_num, nlinear):
        son_list = []
        nlinear_list = []
        skip_linear = []
        
        queue = [(0,0)]
        head = 0
        tot_linear = 0
        while head<len(queue):
            now, depth = queue[head]
            if depth==2:
                skip_linear.append(tot_linear)
            nlinear_list.append(nlinear[depth])
            tot_linear+=nlinear[depth]
            head+=1
            son = []
            for i in range(son_num[depth]):
                next = len(queue)
                queue.append((next, depth+1))
                son.append(next)
            son_list.append(son)
        print("son_list",son_list)
        print("nlinear_list",nlinear_list)
        print("skip_linear",skip_linear)

        return son_list, nlinear_list, skip_linear

    def build_tree(self, head_num):
        # self.son_list = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # self.nlinear_list = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

        # self.son_list = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[],[],[],[],[],[],[],[]]
        # self.nlinear_list = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

        # self.son_list = [[1,2],[3,4],[5,6],[],[],[],[]]
        # self.nlinear_list = [2,2,2,4,4,4,4]
        # self.skip_linear = [6,10,14,18]

        if head_num == 1:
            # 1 head
            self.son_list = [[1],[2],[3],[]]
            self.nlinear_list = [2,2,4,4]
            self.skip_linear = [4]
        elif head_num == 0:
            # 1 head
            self.son_list = [[]]
            self.nlinear_list = [8]
            self.skip_linear = [4]
        elif head_num == 2:
            # 1 head
            self.son_list = [[1],[2],[]]
            self.nlinear_list = [2,2,4]
            self.skip_linear = [4]
        elif head_num == 4:
            # 4 head
            self.son_list = [[1,2],[3,4],[5,6],[7],[8],[9],[10],[],[],[],[]]
            self.nlinear_list = [2,2,2,4,4,4,4,4,4,4,4]
            self.skip_linear = [6,10,14,18]
        elif head_num == 7:
            # 8 head, 8max depth
            self.son_list = [[1,2,3,4],[5,6],[7,8],[9,10],[11,12],[],[],[],[],[],[],[],[]]
            self.nlinear_list = [2,2,2,2,2,4,4,4,4,4,4,4,4]
            self.skip_linear = [10,14,18,22,26,30,34,38]
        elif head_num == 8:
            # 8 head
            self.son_list = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[],[],[],[],[],[],[],[]]
            self.nlinear_list = [2,2,2,4,4,4,4,4,4,4,4,4,4,4,4]
            self.skip_linear = [6,10,14,18]
        elif head_num == 16:
            # 16 head
            # self.son_list = [[1,2,3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24],[25,26],[27,28],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
            # self.nlinear_list = [2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
            # self.skip_linear = [10,14,18,22,26,30,34,38]
            son_num = [4,2,2,0]
            nlinear = [2,2,4,4]
            self.son_list, self.nlinear_list, self.skip_linear = self.get_son_list(son_num, nlinear)
        elif head_num == 32:
            son_num = [4,4,2,0]
            nlinear = [2,2,4,4]
            self.son_list, self.nlinear_list, self.skip_linear = self.get_son_list(son_num, nlinear)
        elif head_num == 64:
            son_num = [4,4,4,0]
            nlinear = [2,2,4,4]
            # son_num = [4,2,2,0]
            # nlinear = [2,2,4,4]
            self.son_list, self.nlinear_list, self.skip_linear = self.get_son_list(son_num, nlinear)
            # exit(0)

        # self.anchor_list = [np.array([[-2,0,0],[2,0,0]]).astype(np.float32),
        #     np.array([[0,-2,0],[0,2,0]]).astype(np.float32),
        #     np.array([[0,-2,0],[0,2,0]]).astype(np.float32),
        #     np.array([[-1,0,0],[0,0,0]]).astype(np.float32),
        #     np.array([[-1,0,0],[0,0,0]]).astype(np.float32),
        #     np.array([[0,0,0],[1,0,0]]).astype(np.float32),
        #     np.array([[0,0,0],[1,0,0]]).astype(np.float32)]
        # self.anchor_list = [np.array([[-2,0,0],[2,0,0]]).astype(np.float32),
        #     np.array([[0,-2,0],[0,2,0]]).astype(np.float32),
        #     np.array([[0,-2,0],[0,2,0]]).astype(np.float32),
        #     np.array([[0,0,0]]).astype(np.float32),
        #     np.array([[0,0,0]]).astype(np.float32),
        #     np.array([[0,0,0]]).astype(np.float32),
        #     np.array([[0,0,0]]).astype(np.float32)]
        assert len(self.son_list) == len(self.nlinear_list)
        self.anchor_list = [np.array([[0,0]]).astype(np.float32)]*len(self.son_list)
        self.node_list = [] 
        self.node_num = len(self.son_list)
        self.anchor_num = 0
        self.linear_num = 0
        for i in range(len(self.son_list)):
            son = self.son_list[i]
            if len(son)>0:
                anchor = "anchor"+str(self.anchor_num)
                self.anchor_num += 1
                setattr(self, anchor, jt.array(self.anchor_list[i]))
                # setattr(self, anchor, jt.random([len(son), 3]))
            else:
                anchor = None
            linear = list(range(self.linear_num, self.linear_num+self.nlinear_list[i]))
            self.linear_num += self.nlinear_list[i]
            self.node_list.append(Node(anchor, son, linear))
    
    def my_concat(self, a, b, dim):
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return jt.concat([a,b],dim)

    def search(self, t, p, h, input_pts, input_views, remain_mask):
        node = self.node_list[t]
        # print("search t",t,"remain_mask",remain_mask.sum())
        identity = h
        for i in range(len(node.linears)):
            # print("i",i)
            # print("h",h.shape)
            # print(self.pts_linears[node.linears[i]])
            # print("len",len(self.pts_linears),"node.linears[i]",node.linears[i],"node",node)
            # print("t",t,"i",i,"h",h.shape,"line",self.pts_linears[node.linears[i]].weight.shape)
            h = self.pts_linears[node.linears[i]](h)
            # if t==0 and i==0:
            #     identity = h
            # if i==len(node.linears)-1:
            #     h = h+identity
            if i%2==1:
                h += identity
            h = jt.nn.relu(h)
            if i%2==1 or (t==0 and i==0):
                identity = h
            if node.linears[i] in self.skip_linear:
                h = jt.concat([input_pts, h], -1)

        confidence = self.confidence_linears[t](h).view(-1)
        # threshold = 0.0
        threshold = self.threshold
        # threshold = -1e10
        output = self.outnet[t](h, input_views)
        # output = self.outnet[0](h, input_views)
        out_num = np.zeros((self.node_num))
        
        if len(node.sons)>0 and (not t in self.force_out):
            son_outputs = None
            son_outputs_fuse = None
            son_confs = None
            son_confs_fuse = None
            idxs = None
            idxs_fuse = None

            anchor = self.get_anchor(node.anchors)
            dis = (anchor.unsqueeze(0)-p.unsqueeze(1)).sqr().sum(-1).sqrt()
            min_idx, _ = jt.argmin(dis,-1)
            for i in range(len(node.sons)):
                # print("t",t,"i",i)
                next_t = node.sons[i]
                sidx = jt.arange(0,p.shape[0])
                # print("min_idx==i",min_idx==i)
                sidx = sidx[min_idx==i]
                # print("sidx",sidx)
                next_p = p[sidx]
                next_h = h[sidx]
                next_input_pts = input_pts[sidx]
                next_input_views = input_views[sidx]
                next_remain_mask = remain_mask[sidx].copy()
                next_conf = confidence[sidx]
                next_remain_mask[threshold>next_conf] = 0
                sidx_fuse = sidx[next_remain_mask==1]
                
                # print("start t",t,"i",i,"next_t",next_t)
                next_outputs, next_outputs_fuse, next_confs, next_confs_fuse, next_out_num = self.search(next_t, next_p, next_h, next_input_pts, next_input_views, next_remain_mask)
                out_num = out_num+next_out_num
                # print("search", t, next_t)
                # print("next_outputs",next_outputs.shape)
                # print("next_outputs_fuse",next_outputs_fuse.shape)
                # print("next_confs",next_confs.shape)
                # print("next_confs_fuse",next_confs_fuse.shape)

                # print("end t",t,"i",i,"next_t",next_t)
                son_outputs = self.my_concat(son_outputs, next_outputs, 1)
                son_outputs_fuse = self.my_concat(son_outputs_fuse, next_outputs_fuse, 1)
                son_confs = self.my_concat(son_confs, next_confs, 1)
                son_confs_fuse = self.my_concat(son_confs_fuse, next_confs_fuse, 1)
                idxs = self.my_concat(idxs, sidx, 0)
                idxs_fuse = self.my_concat(idxs_fuse, sidx_fuse, 0)
            
            # print("t",t)
            son_outputs_save = jt.zeros(son_outputs.shape) 
            son_outputs_save[:,idxs] = son_outputs
            son_outputs_save = jt.concat([output.unsqueeze(0), son_outputs_save], 0)
            son_confs_save = jt.zeros(son_confs.shape) 
            son_confs_save[:,idxs] = son_confs
            son_confs_save = jt.concat([confidence.unsqueeze(1).unsqueeze(0), son_confs_save], 0)

            out_remain_mask = remain_mask.copy()
            out_remain_mask[threshold<=confidence] = 0
            idx_out = jt.arange(0,out_remain_mask.shape[0])[out_remain_mask==1]
            outputs_out = output[idx_out].unsqueeze(0)
            out_num[t] = outputs_out.shape[1]
            confs_out = confidence[idx_out].unsqueeze(1).unsqueeze(0)
            outputs_out = jt.concat([outputs_out, son_outputs_fuse], 1)
            confs_out = jt.concat([confs_out, son_confs_fuse], 1)
            idx_out = jt.concat([idx_out, idxs_fuse], 0)
            
            outputs_out_save = jt.zeros(output.unsqueeze(0).shape)
            outputs_out_save[:, idx_out] = outputs_out
            outputs_out_save  = outputs_out_save[:, remain_mask==1]
            confs_out_save = jt.zeros(confidence.unsqueeze(1).unsqueeze(0).shape) 
            confs_out_save[:, idx_out] = confs_out
            confs_out_save  = confs_out_save[:, remain_mask==1]

            return son_outputs_save, outputs_out_save, son_confs_save, confs_out_save, out_num
        else:
            outputs_save = output.unsqueeze(0)
            outputs_save_log = outputs_save.copy()
            confs_save = confidence.unsqueeze(1).unsqueeze(0)
            # print("outputs_save",outputs_save.shape)
            # print("remain_mask",remain_mask.shape)
            # print("remain_mask==1",remain_mask==1)
            outputs_out_save = outputs_save[:, remain_mask==1]
            confs_out_save = confs_save[:, remain_mask==1]
            out_num[t] = outputs_out_save.shape[1]
            
            remain_mask[threshold<=confidence] = 0
            # print("out:", remain_mask.sum().numpy(), "remain:", remain_mask.shape[0]-remain_mask.sum().numpy())

            # print("outputs_out_save",outputs_out_save.shape)
            if not self.training:
                if t%4==0:
                    outputs_save_log[..., 0] *= 0.
                    outputs_save_log[..., 1] *= 0.
                elif t%4==1:
                    outputs_save_log[..., 0] *= 0.
                    outputs_save_log[..., 2] *= 0.
                elif t%4==2:
                    outputs_save_log[..., 1] *= 0.
                    outputs_save_log[..., 2] *= 0.
                elif t%4==3:
                    outputs_save_log[..., 0] *= 0.
                # elif t==1:
                #     outputs_out_save[..., 1] *= 0.
                # elif t==2:
                #     outputs_out_save[..., 2] *= 0.
                outputs_save = jt.concat([outputs_save, outputs_save_log], 0)
                confs_save = jt.concat([confs_save, confs_save], 0)
            # print("outputs_out_save out",outputs_out_save.shape)

            # print("outputs_out_save",outputs_out_save.shape)

            return outputs_save, outputs_out_save, confs_save, confs_out_save, out_num

    def do_train(self, x, p):
        input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        remain_mask = jt.ones(input_pts.shape[0])
        outputs, outputs_fuse, confs, confs_fuse, out_num = self.search(0, p, input_pts, input_pts, input_views, remain_mask)

        outputs = jt.concat([outputs, outputs_fuse], 0)
        confs = jt.concat([confs, confs_fuse], 0)

        return outputs, confs, np.zeros([1])

    def do_eval(self, x, p):
        input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        remain_mask = jt.ones(input_pts.shape[0])
        outputs, outputs_fuse, confs, confs_fuse, out_num = self.search(0, p, input_pts, input_pts, input_views, remain_mask)

        log = "out: "
        sout_num = list(out_num)
        for i in range(len(sout_num)):
            log += str(i)+": %d;  " % sout_num[i]
        # print(log)
        sout_num = np.array(sout_num)
        outputs = outputs[-1:]

        outputs = jt.concat([outputs, outputs_fuse], 0)
        confs = jt.concat([confs, confs_fuse], 0)

        return outputs, confs, sout_num
        
    def execute(self, x, p, training):
        self.training = training
        if training:
            return self.do_train(x, p)
        else:
            # return self.do_train(x, p)
            return self.do_eval(x, p)

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = jt.array(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = jt.array(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = jt.array(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = jt.array(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = jt.array(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = jt.array(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = jt.array(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = jt.array(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = jt.array(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = jt.array(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, focal, c2w, intrinsic = None):
    i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    if intrinsic is None:
        dirs = jt.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jt.ones_like(i)], -1).unsqueeze(-2)
    else:
        i+=0.5
        j+=0.5
        dirs = jt.stack([i, j, jt.ones_like(i)], -1).unsqueeze(-2)
        dirs = jt.sum(dirs * intrinsic[:3,:3], -1).unsqueeze(-2)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.sum(dirs * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t.unsqueeze(-1) * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = jt.stack([o0,o1,o2], -1)
    rays_d = jt.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.random(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        # np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = jt.array(u)

    # Invert CDF
    # u = u.contiguous()
    # inds = searchsorted(cdf, u, side='right')
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    cond = jt.where(denom<1e-5)
    denom[cond] = 1.
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
