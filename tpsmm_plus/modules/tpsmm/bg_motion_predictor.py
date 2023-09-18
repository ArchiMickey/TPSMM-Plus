from torch import nn
import torch
from torchvision import models

from tpsmm_plus.modules.tpsmm.util import AntiAliasInterpolation2d
from tpsmm_plus.modules.simple_vit import SimpleViT

class BGMotionPredictor(nn.Module):
    """
    Module for background estimation, return single transformation, parametrized as 3x3 matrix. The third row is [0 0 1]
    """

    def __init__(self, scale_factor=1, **kwargs):
        super(BGMotionPredictor, self).__init__()
        
        self.scale_factor = scale_factor
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, scale_factor)
        
        self.bg_encoder = models.resnet18(pretrained=False)
        self.bg_encoder.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.bg_encoder.fc.in_features
        self.bg_encoder.fc = nn.Linear(num_features, 6)
        self.bg_encoder.fc.weight.data.zero_()
        self.bg_encoder.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.bg_encoder = SimpleViT(
        #     image_size=256,
        #     patch_size=16,
        #     num_classes=6,
        #     dim=256,
        #     depth=6,
        #     heads=8,
        #     mlp_dim=512,
        #     channels=6,
        # )
        # self.bg_encoder.to_patch_embedding[2] = nn.Linear(6*16*16, 256, bias=False)
        # self.bg_encoder.linear_head[1].weight.data.zero_()
        # self.bg_encoder.linear_head[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, source_image, driving_image):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
            driving_image = self.down(driving_image)
        
        bs = source_image.shape[0]
        out = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).type(source_image.type())
        prediction = self.bg_encoder(torch.cat([source_image, driving_image], dim=1))
        out[:, :2, :] = prediction.view(bs, 2, 3)
        return out

if __name__ == "__main__":
    model = BGMotionPredictor()
    source_image = torch.randn(1, 3, 256, 256)
    driving_image = torch.randn(1, 3, 256, 256)
    out = model(source_image, driving_image)
    print(out)
    print(out.shape)
    
    vit = SimpleViT(
        image_size=256,
        patch_size=16,
        num_classes=6,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        channels=6,
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in vit.parameters() if p.requires_grad))