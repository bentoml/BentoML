# --------------------------------------------------------------------------------
# VIT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# CoaT: https://github.com/mlpc-ucsd/CoaT
# --------------------------------------------------------------------------------


from detectron2.layers import ShapeSpec
from detectron2.modeling import FPN
from detectron2.modeling import Backbone
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .beit import dit_base_patch16
from .beit import beit_base_patch16
from .beit import dit_large_patch16
from .beit import beit_large_patch16
from .deit import mae_base_patch16
from .deit import deit_base_patch16

__all__ = ["build_vit_fpn_backbone"]


class VIT_Backbone(Backbone):
    """
    Implement VIT backbone.
    """

    def __init__(self, name, out_features, drop_path, img_size, pos_type, model_kwargs):
        super().__init__()
        self._out_features = out_features
        if "base" in name:
            self._out_feature_strides = {
                "layer3": 4,
                "layer5": 8,
                "layer7": 16,
                "layer11": 32,
            }
        else:
            self._out_feature_strides = {
                "layer7": 4,
                "layer11": 8,
                "layer15": 16,
                "layer23": 32,
            }

        if name == "beit_base_patch16":
            model_func = beit_base_patch16
            self._out_feature_channels = {
                "layer3": 768,
                "layer5": 768,
                "layer7": 768,
                "layer11": 768,
            }
        elif name == "dit_base_patch16":
            model_func = dit_base_patch16
            self._out_feature_channels = {
                "layer3": 768,
                "layer5": 768,
                "layer7": 768,
                "layer11": 768,
            }
        elif name == "deit_base_patch16":
            model_func = deit_base_patch16
            self._out_feature_channels = {
                "layer3": 768,
                "layer5": 768,
                "layer7": 768,
                "layer11": 768,
            }
        elif name == "mae_base_patch16":
            model_func = mae_base_patch16
            self._out_feature_channels = {
                "layer3": 768,
                "layer5": 768,
                "layer7": 768,
                "layer11": 768,
            }
        elif name == "dit_large_patch16":
            model_func = dit_large_patch16
            self._out_feature_channels = {
                "layer7": 1024,
                "layer11": 1024,
                "layer15": 1024,
                "layer23": 1024,
            }
        elif name == "beit_large_patch16":
            model_func = beit_large_patch16
            self._out_feature_channels = {
                "layer7": 1024,
                "layer11": 1024,
                "layer15": 1024,
                "layer23": 1024,
            }
        else:
            raise ValueError("Unsupported VIT name yet.")

        if "beit" in name or "dit" in name:
            if pos_type == "abs":
                self.backbone = model_func(
                    img_size=img_size,
                    out_features=out_features,
                    drop_path_rate=drop_path,
                    use_abs_pos_emb=True,
                    **model_kwargs,
                )
            elif pos_type == "shared_rel":
                self.backbone = model_func(
                    img_size=img_size,
                    out_features=out_features,
                    drop_path_rate=drop_path,
                    use_shared_rel_pos_bias=True,
                    **model_kwargs,
                )
            elif pos_type == "rel":
                self.backbone = model_func(
                    img_size=img_size,
                    out_features=out_features,
                    drop_path_rate=drop_path,
                    use_rel_pos_bias=True,
                    **model_kwargs,
                )
            else:
                raise ValueError()
        else:
            self.backbone = model_func(
                img_size=img_size,
                out_features=out_features,
                drop_path_rate=drop_path,
                **model_kwargs,
            )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"VIT takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        return self.backbone.forward_features(x)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


def build_VIT_backbone(cfg):
    """
    Create a VIT instance from config.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        A VIT backbone instance.
    """
    # fmt: off
    name = cfg.MODEL.VIT.NAME
    out_features = cfg.MODEL.VIT.OUT_FEATURES
    drop_path = cfg.MODEL.VIT.DROP_PATH
    img_size = cfg.MODEL.VIT.IMG_SIZE
    pos_type = cfg.MODEL.VIT.POS_TYPE

    model_kwargs = eval(str(cfg.MODEL.VIT.MODEL_KWARGS).replace("`", ""))

    return VIT_Backbone(name, out_features, drop_path, img_size, pos_type, model_kwargs)


@BACKBONE_REGISTRY.register()
def build_vit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a VIT w/ FPN backbone.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_VIT_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
