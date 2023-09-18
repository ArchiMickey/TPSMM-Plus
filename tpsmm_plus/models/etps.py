import lightning.pytorch as pl
import wandb
import torch

torch.set_float32_matmul_precision("medium")

from tpsmm_plus.modules.tpsmm.dense_motion import DenseMotionNetwork
from tpsmm_plus.modules.tpsmm.inpainting_network import InpaintingNetwork
from tpsmm_plus.modules.tpsmm.keypoint_detector import KPDetector
from tpsmm_plus.modules.tpsmm.bg_motion_predictor import BGMotionPredictor
from tpsmm_plus.modules.losses.etps_loss import ETPSLoss
from util import Plotter


class ETPS(pl.LightningModule):
    def __init__(
        self,
        model_params,
        train_params,
        loss_params,
        transform_params,
        visualizer_params,
    ):
        super().__init__()

        self.kp_detector = KPDetector(
            **model_params["kp_detector_params"], **model_params["common_params"]
        )

        self.inpainting_network = InpaintingNetwork(
            **model_params["generator_params"], **model_params["common_params"]
        )

        self.dense_motion_network = DenseMotionNetwork(
            **model_params["dense_motion_params"], **model_params["common_params"]
        )

        self.bg_predictor = BGMotionPredictor(
            **model_params["bg_predictor_params"], **model_params["common_params"]
        )

        self.loss_fn = ETPSLoss(**loss_params)

        self.train_params = train_params
        self.loss_params = loss_params
        self.transform_params = transform_params

        self.bg_start = train_params["bg_start"]

        # Dropout params
        self.dropout_epoch = train_params["dropout_epoch"]
        self.dropout_maxp = train_params["dropout_maxp"]
        self.dropout_inc_epoch = train_params["dropout_inc_epoch"]
        self.dropout_startp = train_params["dropout_startp"]
        
        self.plotter = Plotter(**visualizer_params)

    def forward(self, source_imgs, driving_img):
        return

    def general_step(self, batch, batch_idx, is_train=True):
        out_dict = self.default_kwargs()

        src = batch["source"]
        driving = batch["driving"]
        out_dict |= {"source": src, "driving": driving}

        kp_src = self.kp_detector(src)
        kp_driving = self.kp_detector(driving)
        out_dict |= {"kp_source": kp_src, "kp_driving": kp_driving}

        bg_param = None
        if self.current_epoch >= self.bg_start:
            bg_param = self.bg_predictor(src, driving)
        out_dict |= {"bg_param": bg_param}

        dropout_flag, dropout_p = self.get_dropout_params(is_train)
        dense_motion = self.dense_motion_network(
            source_image=src,
            kp_driving=kp_driving,
            kp_source=kp_src,
            bg_param=bg_param,
            dropout_flag=dropout_flag,
            dropout_p=dropout_p,
        )
        out_dict |= dense_motion

        generated = self.inpainting_network(src, dense_motion, kp_src, kp_driving)
        out_dict |= {"generated": generated}

        loss, loss_dict = self.loss_fn(**out_dict)

        out_dict |= loss_dict

        return loss, out_dict

    def training_step(self, batch, batch_idx):
        loss, out_dict = self.general_step(batch, batch_idx)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        log_dict = {}
        for loss_name in self.loss_params["loss_weights"].keys():
            log_dict[f"train/{loss_name}_loss"] = out_dict[f"{loss_name}_loss"]
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, out_dict = self.general_step(batch, batch_idx, is_train=False)

        self.log("val/loss", loss, on_epoch=True, logger=True)

        log_dict = {}
        for loss_name in self.loss_params["loss_weights"].keys():
            log_dict[f"val/{loss_name}_loss"] = out_dict[f"{loss_name}_loss"]

        self.log_dict(log_dict, on_epoch=True, logger=True)

        if batch_idx == 0:
            self.log_generated_driving(out_dict)

        return loss

    def configure_optimizers(self):
        optimizer = self.train_params["optimizer"](
            params=list(self.inpainting_network.parameters())
            + list(self.kp_detector.parameters())
            + list(self.dense_motion_network.parameters())
            + list(self.bg_predictor.parameters())
        )
        scheduler = self.train_params["lr_scheduler"](optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

    def get_dropout_params(self, is_train=True):
        if is_train or self.current_epoch >= self.dropout_epoch:
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp
            dropout_flag = True
            dropout_p = min(
                self.current_epoch / self.dropout_inc_epoch * self.dropout_maxp
                + self.dropout_startp,
                self.dropout_maxp,
            )
        return dropout_flag, dropout_p

    def log_generated_driving(self, out_dict):
        grid = self.plotter.create_image_grid(**out_dict)
        wandb.log(
            {"img/gen_driving": wandb.Image(grid), "global_step": self.global_step}
        )

    def default_kwargs(self):
        return {
            "inpainting_network": self.inpainting_network,
            "kp_extractor": self.kp_detector,
            "bg_predictor": self.bg_predictor,
            "transform_params": self.transform_params,
        }
