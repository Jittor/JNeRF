import functools
from typing import Any, Callable
import jittor as jt
from jittor import nn
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
from jnerf.utils.miputils import *


@NETWORKS.register_module()
class MipNerfMLP1(nn.Module):
    def __init__(self):
        super(MipNerfMLP1, self).__init__()
        self.cfg = get_cfg()
        feature_dim = (self.cfg.max_deg_point - self.cfg.min_deg_point) * 6
        self.num_samples = self.cfg.num_samples
        self.net_depth = self.cfg.net_depth
        self.skip_layer = self.cfg.skip_layer
        self.first_part = nn.Sequential()
        net_depth = self.cfg.net_depth
        net_width = self.cfg.net_width
        skip_layer = self.cfg.skip_layer
        self.num_density_channels = self.cfg.num_density_channels
        self.num_rgb_channels = self.cfg.num_rgb_channels
        net_width_condition = self.cfg.net_width_condition
        net_depth_condition = self.cfg.net_depth_condition
        self.first_part.append(nn.Linear(feature_dim, net_width))  # relu在forward写了
        for _ in range(1, net_depth):
            if _ % skip_layer == 1 and _ != 1:
                if _ == skip_layer + 1:
                    self.first_part.append(nn.Linear(net_width + feature_dim, net_width))
                else:
                    self.first_part.append(nn.Linear(net_width * 2, net_width))
            else:
                self.first_part.append(nn.Linear(net_width, net_width))
        self.density_layer = nn.Linear(net_width, self.num_density_channels)
        self.condition = nn.Sequential()
        self.bottleneck = nn.Linear(net_width, net_width)
        if net_depth_condition >= 1:
            self.condition.append(nn.Linear(net_width + 27, net_width_condition))
            self.condition.append(nn.ReLU())
        for _ in range(1, net_depth_condition):
            self.condition.append(nn.Linear(net_width_condition, net_width_condition))
            self.condition.append(nn.ReLU())
        self.rgb_layer = nn.Linear(net_width, self.num_rgb_channels)
        self.rgb_condition = nn.Linear(net_width_condition, self.num_rgb_channels)
        self.using_fp16 = self.cfg.using_fp16
        if self.using_fp16:
            self.set_fp16()

    def execute(self, inputs, condition=None):  # inputs:(batch_size,7) samples_enc, viewdirs_enc
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(inputs, condition)
        else:
            return self.execute_(inputs, condition)

    def execute_(self, x, condition=None):
        """Evaluate the MLP.

        Args:
        x: jnp.ndarray(float32), [batch, num_samples, feature], points.
        condition: jnp.ndarray(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
        raw_rgb: jnp.ndarray(float32), with a shape of
            [batch, num_samples, num_rgb_channels].
        raw_density: jnp.ndarray(float32), with a shape of
            [batch, num_samples, num_density_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.reshape([-1, feature_dim])
        # dense_layer = functools.partial(
        #     nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
        inputs = x
        for i in range(self.net_depth):
            x = jt.nn.relu(self.first_part[i](x))
            if i % self.skip_layer == 0 and i > 0:
                x = jt.concat([x, inputs], -1)

        raw_density = self.density_layer(x).reshape(
            [-1, num_samples, self.num_density_channels])

        if condition is not None:
            # Output of the first part of MLP.
            bottleneck = self.bottleneck(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            condition = condition[:, None, :].repeat(1, num_samples, 1)
            # condition = jt.tile(condition[:, None, :], (1, num_samples, 1))
            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.reshape([-1, condition.shape[-1]])
            x = jt.concat([bottleneck, condition], -1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.condition(x)
            raw_rgb = self.rgb_condition(x).reshape((-1, self.num_samples, self.num_rgb_channels))
        else:
            raw_rgb = self.rgb_layer(x).reshape((-1, self.num_samples, self.num_rgb_channels))
        return raw_rgb, raw_density

    def set_fp16(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.float16()

    # def weight_init(self):
    #     from jittor import init
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             init.constant_(m.weight, 0.01)
    #             init.constant_(m.bias, 0)


def _xavier_init(linear):
    nn.init.xavier_uniform(linear.weight.data)

@NETWORKS.register_module()
class MipNerfMLP(nn.Module):
    def __init__(self):
        super(MipNerfMLP, self).__init__()
        self.cfg = get_cfg()
        net_depth = self.cfg.net_depth
        net_width = self.cfg.net_width
        self.skip_index = self.cfg.skip_layer
        net_width_condition = self.cfg.net_width_condition
        num_rgb_channels = self.cfg.num_rgb_channels
        num_density_channels = self.cfg.num_density_channels
        net_depth_condition = self.cfg.net_depth_condition
        layers = []
        feature_dim = (self.cfg.max_deg_point - self.cfg.min_deg_point) * 6
        view_dim = self.cfg.deg_view * 3 * 2
        view_dim = view_dim + 3  # 4 * 6 + 3 = 27
        for i in range(net_depth):
            if i == 0:
                dim_in = feature_dim
                dim_out = net_width
            elif (i - 1) % self.skip_index == 0 and i > 1:
                dim_in = net_width + feature_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = nn.Linear(dim_in, dim_out)
            # _xavier_init(linear)
            layers.append(nn.Sequential(linear, nn.ReLU()))
        self.layers = nn.ModuleList(layers)
        del layers
        self.density_layer = nn.Linear(net_width, num_density_channels)
        # _xavier_init(self.density_layer)
        self.extra_layer = nn.Linear(net_width, net_width)
        # _xavier_init(self.extra_layer)

        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = nn.Linear(dim_in, dim_out)
            # _xavier_init(linear)
            layers.append(nn.Sequential(linear, nn.ReLU()))
        self.view_layers = nn.Sequential(*layers)
        del layers
        self.color_layer = nn.Linear(net_width_condition, num_rgb_channels)

        self.using_fp16 = self.cfg.using_fp16
        if self.using_fp16:
            self.set_fp16()

    def execute(self, inputs, condition=None):  # inputs:(batch_size,7) samples_enc, viewdirs_enc
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(inputs, condition)
        else:
            return self.execute_(inputs, condition)

    def execute_(self, x, condition=None):
        """Evaluate the MLP.

        Args:
        x: jnp.ndarray(float32), [batch, num_samples, feature], points.
        condition: jnp.ndarray(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
        raw_rgb: jnp.ndarray(float32), with a shape of
            [batch, num_samples, num_rgb_channels].
        raw_density: jnp.ndarray(float32), with a shape of
            [batch, num_samples, num_density_channels].
        """
        num_samples = x.shape[1]
        inputs = x   # [B, N, 2*3*L]， N是num_samples
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = jt.concat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)
        if condition is not None:
            bottleneck = self.extra_layer(x)
            condition = condition[:, None, :].repeat(1, num_samples, 1)
            x = jt.concat([bottleneck, condition], dim=-1)
            x = self.view_layers(x)
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density

    def set_fp16(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.float16()

