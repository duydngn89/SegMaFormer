import os

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import (
    DC_and_BCE_loss,
    DC_and_CE_loss,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.network_architecture.SegMaFormer import (
    build_segmaformer_model_from_yaml,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class ACDCSegMaFormerTrainerBase(nnUNetTrainer):
    config_filename = "ACDC_config.yml"
    deep_supervision_weights = np.array([1.0, 0.25, 0.125, 0.125, 0.125], dtype=np.float32)

    @classmethod
    def build_network_architecture(
        cls, network_arch_class_name, network_arch_init_kwargs,
        network_arch_init_kwargs_req_import, num_input_channels, num_classes,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        config_path = os.path.join(os.path.dirname(__file__), "network_architecture",
                                   "configs", "ACDC_config.yml")
        return build_segmaformer_model_from_yaml(config_path, overrides={
            "in_channels": num_input_channels,
            "num_classes": num_classes,
        })

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                weight_ce=0.2,
                weight_dice=0.8,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = self._build_multiclass_loss()

        if self._do_i_compile():
            try:
                loss = torch.compile(loss)
                print("[SegMaFormer] Compiled loss with torch.compile()")
            except Exception as exc:
                print(f"[SegMaFormer] torch.compile() failed for loss: {exc}")

        if self.enable_deep_supervision:
            weights = self.deep_supervision_weights.astype(np.float32, copy=True)
            weights /= weights.sum()
            print(f"[SegMaFormer] Deep supervision weights: {weights.tolist()}")
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _build_multiclass_loss(self):
        return DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
