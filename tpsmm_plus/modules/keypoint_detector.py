from torch import nn
import torch
from torchvision import models
import torch.nn.functional as F

from tpsmm_plus.modules.util import (
    Hourglass,
    AntiAliasInterpolation2d,
    make_coordinate_grid,
    DownBlock2d,
)
from tpsmm_plus.modules.convnextv2.convnextv2 import convnextv2_femto


class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(
        self, num_tps, scale_factor=1, mode="resnet", temperature=0.1, **kwargs
    ):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps
        self.scale_factor = scale_factor
        self.mode = mode
        self.temperature = temperature

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, scale_factor)

        if mode == "resnet":
            self.fg_encoder = models.resnet18(pretrained=False)
            num_features = self.fg_encoder.fc.in_features
            self.fg_encoder.fc = nn.Linear(num_features, num_tps * 5 * 2)

        elif mode == "convnext":
            self.fg_encoder = convnextv2_femto(num_classes=num_tps * 5 * 2)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = (
            make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        )
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = value

        return kp

    def resnet_forward(self, image):
        fg_kp = self.fg_encoder(image)
        (
            bs,
            _,
        ) = fg_kp.shape
        fg_kp = torch.tanh(fg_kp)
        out = {"fg_kp": fg_kp.view(bs, self.num_tps * 5, -1)}
        return out

    def forward(self, image):
        if self.scale_factor != 1:
            image = self.down(image)

        if self.mode in ["resnet", "convnext"]:
            return self.resnet_forward(image)


class UpBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.act = nn.SiLU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.act(x)
        return x


class KPDetectorWithDepth(nn.Module):
    def __init__(self, num_tps, scale_factor=1, **kwargs):
        super(KPDetectorWithDepth, self).__init__()
        self.num_tps = num_tps
        self.scale_factor = scale_factor

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, scale_factor)

        self.fg_encoder = convnextv2_femto(num_classes=num_tps * 5 * 2)
        self.depth_decoder = nn.ModuleList([])
        ch = [48, 96, 192, 384]
        ch = ch[::-1]
        for i in range(len(ch) - 1):
            if i == 0:
                self.depth_decoder.append(UpBlock(ch[i], ch[i + 1]))
            else:
                self.depth_decoder.append(UpBlock(2 * ch[i], ch[i + 1]))
        self.depth_final = nn.Sequential(
            nn.Conv2d(ch[-1], 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        fg_kp, hs = self.fg_encoder(x, True)
        bs, _ = fg_kp.shape
        fg_kp = torch.tanh(fg_kp)
        out = {"fg_kp": fg_kp.view(bs, self.num_tps * 5, -1)}

        h = hs.pop()
        for i, block in enumerate(self.depth_decoder):
            h = block(h)
            if i != (len(self.depth_decoder) - 1):
                h = torch.cat([h, hs.pop()], dim=1)

        depth = self.depth_final(h)
        out["depth"] = depth

        return out
