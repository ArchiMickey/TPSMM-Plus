import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision import transforms
from torchvision.transforms import Resize

from tpsmm_plus.modules.util import AntiAliasInterpolation2d, TPS
from tpsmm_plus.modules.vit_extractor import VitExtractor


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
    def __init__(self, channels, scales, loss_weights, use_vit_loss=False):
        super().__init__()

        self.channels = channels
        self.scales = scales
        self.pyramid = ImagePyramid(channels, scales)
        self.lpips = lpips.LPIPS(net="vgg")
        self.equivariance_loss = EquivarianceLoss()
        self.loss_weights = loss_weights

        self.extractor = None
        if use_vit_loss:
            self.extractor = VitExtractor("dino_vits16", "cuda")
            imagenet_norm = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
            global_resize_transform = Resize(224, max_size=480)
            self.global_transform = transforms.Compose(
                [global_resize_transform, imagenet_norm]
            )
            self.lambda_app = 1
            self.lambda_str = 0.1
            self.lambda_id = 0.1

    def calculate_structure_ssim_loss(self, outputs, target_imgs):
        loss = 0.0
        for a, b in zip(target_imgs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(
                    a.unsqueeze(0), layer_num=11
                )
            keys_ssim = self.extractor.get_keys_self_sim_from_input(
                b.unsqueeze(0), layer_num=11
            )
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def forward(
        self,
        generated,
        source,
        driving,
        kp_source,
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

        kp_dist_loss = 0.0
        if "kp_distance" in self.loss_weights and self.loss_weights["kp_distance"] != 0:
            bz, num_kp, kp_dim = kp_source["fg_kp"].shape
            sk = kp_source["fg_kp"].unsqueeze(2) - kp_source["fg_kp"].unsqueeze(1)
            dk = kp_driving["fg_kp"].unsqueeze(2) - kp_driving["fg_kp"].unsqueeze(1)
            source_dist_loss = (
                -torch.sign(
                    (
                        torch.sqrt((sk * sk).sum(-1) + 1e-8)
                        + torch.eye(num_kp).cuda() * 0.2
                    )
                    - 0.2
                )
                + 1
            ).mean()
            driving_dist_loss = (
                -torch.sign(
                    (
                        torch.sqrt((dk * dk).sum(-1) + 1e-8)
                        + torch.eye(num_kp).cuda() * 0.2
                    )
                    - 0.2
                )
                + 1
            ).mean()
            # driving_dist_loss = (torch.sign(1-(torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()))+1).mean()
            kp_dist_loss = self.loss_weights["kp_distance"] * (
                source_dist_loss + driving_dist_loss
            )
        out_dict["kp_distance_loss"] = kp_dist_loss
        
        depth_constraint_loss = 0.0
        if "depth_constraint" in self.loss_weights and self.loss_weights["depth_constraint"] != 0:
            depth_generated = kp_extractor(generated["prediction"])["depth"]
            depth_driving = kp_driving["depth"]
            depth_constraint_loss = self.loss_weights["depth_constraint"] * torch.abs(depth_generated - depth_driving).mean()
        out_dict["depth_constraint_loss"] = depth_constraint_loss
        
        struct_loss = 0.0
        if self.extractor is not None:
            struct_loss = self.calculate_structure_ssim_loss(generated["prediction"], driving)
            struct_loss = struct_loss * self.lambda_str
        out_dict["structure_loss"] = struct_loss

        loss = (
            rec_loss
            + p_loss
            + eq_loss
            + warp_loss
            + bg_loss
            + kp_dist_loss
            + depth_constraint_loss
            + struct_loss
        )

        return loss, out_dict
