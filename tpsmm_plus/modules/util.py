from torch import nn
import torch.nn.functional as F
import torch
import math
from einops import rearrange
from tpsmm_plus.modules.dynamic_conv import Dynamic_conv2d
from functools import partial


class TPS:
    """
    TPS transformation, mode 'kp' for Eq(2) in the paper, mode 'random' for equivariance loss.
    """

    def __init__(self, mode, bs, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == "random":
            noise = torch.normal(
                mean=0, std=kwargs["sigma_affine"] * torch.ones([bs, 2, 3])
            )
            self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
            self.control_points = make_coordinate_grid(
                (kwargs["points_tps"], kwargs["points_tps"]), type=noise.type()
            )
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(
                mean=0,
                std=kwargs["sigma_tps"]
                * torch.ones([bs, 1, kwargs["points_tps"] ** 2]),
            )
        elif mode == "kp":
            kp_1 = kwargs["kp_1"]
            kp_2 = kwargs["kp_2"]
            device = kp_1.device
            kp_type = kp_1.type()
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            K = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2)
            K = K**2
            K = K * torch.log(K + 1e-9)

            one1 = (
                torch.ones(self.bs, kp_1.shape[1], kp_1.shape[2], 1)
                .to(device)
                .type(kp_type)
            )
            kp_1p = torch.cat([kp_1, one1], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 3).to(device).type(kp_type)
            P = torch.cat([kp_1p, zero], 2)
            L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
            L = torch.cat([L, P], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 2).to(device).type(kp_type)
            Y = torch.cat([kp_2, zero], 2)
            one = torch.eye(L.shape[2]).expand(L.shape).to(device).type(kp_type) * 0.01
            L = L + one

            param = torch.matmul(torch.inverse(L), Y)
            self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)

            self.control_points = kp_1
            self.control_params = param[:, :, :n, :]
        else:
            raise Exception("Error TPS mode")

    def transform_frame(self, frame):
        grid = (
            make_coordinate_grid(frame.shape[2:], type=frame.type())
            .unsqueeze(0)
            .to(frame.device)
        )
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == "kp":
            shape.insert(1, self.gs)
        grid = self.warp_coordinates(grid).view(*shape)
        return grid

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        control_points = self.control_points.type(coordinates.type()).to(
            coordinates.device
        )
        control_params = self.control_params.type(coordinates.type()).to(
            coordinates.device
        )

        if self.mode == "kp":
            transformed = (
                torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1))
                + theta[:, :, :, 2:]
            )

            distances = coordinates.view(
                coordinates.shape[0], 1, 1, -1, 2
            ) - control_points.view(self.bs, control_points.shape[1], -1, 1, 2)

            distances = distances**2
            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            transformed = transformed.permute(0, 1, 3, 2) + result

        elif self.mode == "random":
            theta = theta.unsqueeze(1)
            transformed = (
                torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1))
                + theta[:, :, :, 2:]
            )
            transformed = transformed.squeeze(-1)
            ances = coordinates.view(
                coordinates.shape[0], -1, 1, 2
            ) - control_points.view(1, 1, -1, 2)
            distances = ances**2

            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        else:
            raise Exception("Error TPS mode")

        return transformed


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """

    coordinate_grid = make_coordinate_grid(spatial_size, kp.type()).to(kp.device)
    number_of_leading_dimensions = len(kp.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = kp.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = kp.shape[:number_of_leading_dimensions] + (1, 1, 2)
    kp = kp.view(*shape)

    mean_sub = coordinate_grid - kp

    out = torch.exp(-0.5 * (mean_sub**2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class Block(nn.Module):
    def __init__(
        self, in_features, out_features, kernel_size, padding, activation="relu"
    ):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(
        self, in_features, out_features, kernel_size, padding, activation="relu"
    ):
        super(ResBlock2d, self).__init__()
        self.block1 = Block(in_features, out_features, kernel_size, padding, activation)
        self.block2 = Block(
            out_features, out_features, kernel_size, padding, activation
        )

        self.res_conv = (
            nn.Conv2d(in_features, out_features, 1)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        out = self.block1(x)

        out = self.block2(out)

        return out + self.res_conv(x)


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, activation="silu"):
        super(UpBlock2d, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()

    def forward(self, x):
        use_bf16 = x.dtype == torch.bfloat16
        if use_bf16:
            x = x.to(torch.float32)

        out = self.up(x)

        if use_bf16:
            out = out.to(torch.bfloat16)

        out = self.conv(out)
        out = self.norm(out)
        out = self.act(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, activation="relu"):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        # self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        # out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.res_conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.silu(out)
        return out + self.res_conv(x)


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        max_features=256,
        activation="relu",
    ):
        super(Encoder, self).__init__()

        down_block_klass = partial(DownBlock2d, activation=activation)
        down_blocks = []
        for i in range(num_blocks):
            is_last = i == num_blocks - 1
            # attn_klass = Attention if is_last else LinearAttention
            in_features = (
                in_features if i == 0 else min(max_features, block_expansion * (2**i))
            )
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            block = [
                down_block_klass(in_features, out_features),
            ]
            down_blocks.append(nn.ModuleList(block))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        # print('encoder:' ,outs[-1].shape)
        # for attn, block in self.down_blocks:
        for block in self.down_blocks:
            out = outs[-1]
            # out = out + attn(out)
            # out = block(out)
            out = block[0](out)
            outs.append(out)
            # print('encoder:' ,outs[-1].shape)
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        max_features=256,
        activation="relu",
    ):
        super(Decoder, self).__init__()

        up_blocks = []
        self.out_channels = []
        up_block_klass = partial(UpBlock2d, activation=activation)
        for i in range(num_blocks)[::-1]:
            # attn_klass = Attention if i == 0 else LinearAttention
            in_filters = (1 if i == num_blocks - 1 else 2) * min(
                max_features, block_expansion * (2 ** (i + 1))
            )
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2**i))
            block = [
                up_block_klass(in_filters, out_filters),
            ]
            up_blocks.append(nn.ModuleList(block))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_channels.append(block_expansion + in_features)

    def forward(self, x, mode=0):
        out = x.pop()
        outs = []
        for block in self.up_blocks:
            out = block[0](out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            outs.append(out)
        if mode == 0:
            return out
        else:
            return outs


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        max_features=256,
        activation="relu",
        **kwargs,
    ):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(
            block_expansion, in_features, num_blocks, max_features, activation
        )
        self.decoder = Decoder(
            block_expansion, in_features, num_blocks, max_features, activation
        )
        self.out_channels = self.decoder.out_channels
        # self.out_filters = self.decoder.out_filters

    def forward(self, x, mode=0):
        return self.decoder(self.encoder(x), mode)


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) ** 2) / (2 * std**2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)

        dtype = out.dtype
        if dtype == torch.bfloat16:
            out = out.to(torch.float32)

        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        if dtype == torch.bfloat16:
            out = out.to(torch.bfloat16)

        return out


def to_homogeneous(coordinates):
    ones_shape = list(coordinates.shape)
    ones_shape[-1] = 1
    ones = torch.ones(ones_shape).type(coordinates.type())

    return torch.cat([coordinates, ones], dim=-1)


def from_homogeneous(coordinates):
    return coordinates[..., :2] / coordinates[..., 2:3]


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNorm2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = super(LayerNorm2d, self).forward(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


class SpaAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(SpaAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1))
        self.norm = LayerNorm2d(dim)
        self.to_qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, 1),
            nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3),
        )
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())

        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.norm(x)
        q, k, v = self.to_qkv(out).chunk(3, dim=1)
        g = self.gate(out)

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads),
            (q, k, v),
        )

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.relu(attn)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = out * g
        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim):
        super(FeedForward, self).__init__()
        self.norm = LayerNorm2d(dim)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim * 4 * 2, 1),
            nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 3, 1, 1, groups=dim * 4 * 2),
        )
        self.linear = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x):
        out = self.norm(x)
        x, gate = self.fc(out).chunk(2, dim=1)
        out = F.gelu(x) * gate
        out = self.linear(out)
        return out


class SpaAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(SpaAttentionBlock, self).__init__()
        self.attn = SpaAttention(dim, num_heads)
        self.fc = FeedForward(dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.fc(x) + x
        return x


class ModulatedConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, style_dim, demodulate=True
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = nn.Linear(style_dim, in_channel)
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


def attn(q, k, v, scale):
    sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale
    attn = sim.softmax(dim=-1)
    out = torch.einsum("b i j, b j d -> b i d", attn, v)
    return out


def linear_attn(q, k, v, scale):
    q = q.softmax(dim=-2)
    k = k.softmax(dim=-1)

    q = q * scale
    context = torch.einsum("b d n, b e n-> b d e", k, v)
    out = torch.einsum("b d e, b d n -> b e n", context, q)
    return out


class CrossAttention(nn.Module):
    def __init__(self, dim, x_dim, context_dim, attn_type="vanilla"):
        super(CrossAttention, self).__init__()
        self.norm_x = LayerNorm2d(x_dim)
        self.norm_context = LayerNorm2d(context_dim)
        self.to_q = nn.Conv2d(x_dim, dim, 1)
        self.to_kv = nn.Conv2d(context_dim, dim * 2, 1)
        self.to_out = nn.Conv2d(dim, x_dim, 1)
        self.scale = dim ** -0.5
        self.attn_type = attn_type

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_norm = self.norm_x(x)
        context_norm = self.norm_context(context)
        q = F.relu(self.to_q(x_norm))
        kv = F.relu(self.to_kv(context_norm))
        k, v = kv.chunk(2, dim=1)
        
        if self.attn_type == "vanilla":
            q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), (q, k, v))
            out = attn(q, k, v, self.scale)
            out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        elif self.attn_type == "linear":
            q, k, v = map(lambda t: rearrange(t, "b c h w -> b c (h w)"), (q, k, v))
            out = linear_attn(q, k, v, self.scale)
            out = rearrange(out, "b c (h w) -> b c h w", h=h, w=w)

        else:
            raise NotImplementedError("Error attn type")

        out = self.to_out(out)

        return out

class MemoryCrossAttention(nn.Module):
    def __init__(self, dim, x_dim, spatial_dim, num_tps, attn_type="vanilla"):
        super(MemoryCrossAttention, self).__init__()
        self.norm_x = LayerNorm2d(x_dim)
        self.norm_memory = LayerNorm2d(dim)
        self.feat_proj = nn.Conv2d(x_dim, dim, 1)
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_style = nn.Sequential(
            nn.Linear(x_dim + num_tps * 5 * 2, 256),
            nn.GELU(),
            nn.Linear(256, 512),
        )
        self.style_conv = ModulatedConv2d(spatial_dim, dim, 3, style_dim=512)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.to_out = nn.Conv2d(dim, x_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scale = dim**-0.5
        self.attn_type = attn_type

    def forward(self, x, memory, kp_source):
        out_dict = {}
        b, _, h, w = x.shape
        x_in = x
        kp = kp_source.view(b, -1).detach()
        style_scale = self.pool(x).view(b, -1)
        style_code = torch.cat((style_scale, kp), dim=-1)
        style = self.to_style(style_code)

        expand_memory = memory.expand(b, -1, -1, -1)
        styled_memory = self.style_conv(expand_memory, style)

        x = self.norm_x(x)
        feat = self.feat_proj(x)
        out_dict["feat"] = feat

        q = F.relu(self.to_q(feat))
        styled_memory = self.norm_memory(styled_memory)
        kv = F.relu(self.to_kv(styled_memory))
        k, v = kv.chunk(2, dim=1)
        out_dict["value"] = v

        if self.attn_type == "vanilla":
            q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), (q, k, v))
            out = attn(q, k, v, self.scale)
            out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        elif self.attn_type == "linear":
            q, k, v = map(lambda t: rearrange(t, "b c h w -> b c (h w)"), (q, k, v))
            out = linear_attn(q, k, v, self.scale)
            out = rearrange(out, "b c (h w) -> b c h w", h=h, w=w)

        else:
            raise NotImplementedError("Error attn type")

        out = self.to_out(out)
        out_dict["out"] = x_in + out

        return out_dict


if __name__ == "__main__":
    from icecream import install

    install()
    model = Hourglass(32, 3, 3, 256)
    x = torch.randn(1, 3, 256, 256)
    ic(model, model(x).shape)
