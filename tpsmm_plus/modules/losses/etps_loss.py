import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from tpsmm_plus.modules.util import AntiAliasInterpolation2d, TPS


class ImagePyramid(nn.Module):
    def __init__(self, channels, scales):
        super().__init__()
        self.downs = nn.ModuleList(
            [AntiAliasInterpolation2d(channels, scale) for scale in scales]
        )

    def forward(self, x):
        out = []
        for down in self.downs:
            out.append(down(x))
        return out


class EquivarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, driving, kp_driving, kp_extractor, transform_params):
        out_dict = {}

        transform_random = TPS(mode="random", bs=driving.shape[0], **transform_params)
        transform_grid = transform_random.transform_frame(driving)
        transformed_frame = F.grid_sample(
            driving, transform_grid, padding_mode="reflection", align_corners=True
        )
        transformed_kp = kp_extractor(transformed_frame)

        out_dict["transformed_frame"] = transformed_frame
        out_dict["transformed_kp"] = transformed_kp

        warped = transform_random.warp_coordinates(transformed_kp["fg_kp"])
        b, n, d = warped.shape
        kp_d = kp_driving["fg_kp"].view(b, -1, n, d)
        kp_d = kp_d[:, 0, :, :]
        loss = torch.abs(kp_d - warped).mean()
        out_dict["loss"] = loss

        return out_dict


class ETPSLoss(nn.Module):
    def __init__(self, channels, scales, loss_weights):
        super().__init__()

        self.channels = channels
        self.scales = scales
        self.pyramid = ImagePyramid(channels, scales)
        self.lpips = lpips.LPIPS(net="vgg")
        self.equivariance_loss = EquivarianceLoss()
        self.loss_weights = loss_weights

    def forward(
        self,
        generated,
        source,
        driving,
        kp_driving,
        inpainting_network,
        kp_extractor,
        bg_predictor,
        transform_params,
        bg_param=None,
        **kwargs
    ):
        out_dict = {}
        loss = 0.0

        # Reconstruction loss
        rec_loss = 0.0
        if self.loss_weights["reconstruction"] != 0:
            rec_loss = torch.abs(generated["prediction"] - driving).mean()
            rec_loss = rec_loss * self.loss_weights["reconstruction"]
        out_dict["reconstruction_loss"] = rec_loss

        # Perceptual loss
        p_loss = 0.0
        generated_pyramid = self.pyramid(generated["prediction"])
        driving_pyramid = self.pyramid(driving)
        for i in range(len(self.scales)):
            y_ = F.tanh(generated_pyramid[i])
            x_ = F.tanh(driving_pyramid[i])
            p_loss += self.lpips(y_, x_)
        p_loss = p_loss.mean() * self.loss_weights["perceptual"]
        out_dict["perceptual_loss"] = p_loss

        # Equivariance loss
        eq_loss = 0.0
        eq_loss_out = self.equivariance_loss(
            driving, kp_driving, kp_extractor, transform_params
        )
        eq_loss = eq_loss_out.pop("loss") * self.loss_weights["equivariance"]
        out_dict["equivariance_loss"] = eq_loss
        out_dict |= eq_loss_out

        # Warp Loss
        warp_loss = 0.0
        if self.loss_weights["warp"] != 0:
            occlusion_map = generated["occlusion_map"]
            encode_map = inpainting_network.get_encode(driving, occlusion_map)
            decode_map = generated["warped_encoder_maps"]
            for i in range(len(encode_map)):
                warp_loss += torch.abs(encode_map[i] - decode_map[-i - 1]).mean()
            warp_loss = warp_loss * self.loss_weights["warp"]
        out_dict["warp_loss"] = warp_loss

        # Background loss
        bg_loss = 0.0
        if self.loss_weights["bg"] != 0 and bg_param is not None:
            bg_param_reverse = bg_predictor(driving, source)
            bg_loss = torch.matmul(bg_param, bg_param_reverse)
            eye = torch.eye(3).view(1, 1, 3, 3).type(bg_loss.type())
            bg_loss = torch.abs(eye - bg_loss).mean()
            bg_loss = bg_loss * self.loss_weights["bg"]
        out_dict["bg_loss"] = bg_loss

        loss = rec_loss + p_loss + eq_loss + warp_loss + bg_loss

        return loss, out_dict
