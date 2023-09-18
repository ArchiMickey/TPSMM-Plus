from torch import nn
import torch
from torchvision import models

from modules.tpsmm.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, DownBlock2d


class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(
        self,
        num_tps,
        scale_factor=1,
        mode="resnet",
        temperature=0.1,
        strength_dim=64,
        strength_dim_mults=[1, 2, 3, 4],
        **kwargs
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

        elif mode == "unet":
            kwargs["in_features"] = kwargs["num_channels"]
            self.predictor = Hourglass(**kwargs)
            self.kp = nn.Conv2d(
                self.predictor.out_channels[-1], num_tps * 5, 7, padding=3
            )   

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

    def unet_forward(self, image):
        fmap = self.predictor(image)
        prediction = self.kp(fmap)

        heatmap = prediction.view(*prediction.shape[:2], -1)
        heatmap = torch.softmax(heatmap / self.temperature, dim=-1)
        heatmap = heatmap.view(*prediction.shape)
        
        # strength = self.strength(fmap)

        out = {"fg_kp": self.gaussian2kp(heatmap)}
        return out

    def forward(self, image):
        if self.scale_factor != 1:
            image = self.down(image)

        if self.mode == "resnet":
            return self.resnet_forward(image)

        elif self.mode == "unet":
            return self.unet_forward(image)
