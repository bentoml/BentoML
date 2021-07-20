# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import os
import typing as t

import torch

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import detectron2
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
except ImportError:
    raise MissingDependencyException("detectron2 is required by DetectronModel")


class DetectronModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`detectron2` models,
    in the form of :class:`~detectron2.checkpoint.DetectionCheckpointer`

    Args:
        model (`torch.nn.Module`):
            detectron2 model is of type :obj:`torch.nn.Module`
        input_model_yaml (`str`, `optional`, default to `None`):
            model config from :meth:`detectron2.model_zoo.get_config_file`
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`detectron2` is required by DetectronModel
        InvalidArgument:
            model is not an instance of :class:`torch.nn.Module`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    _model: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        input_model_yaml: str = None,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(DetectronModel, self).__init__(model, metadata=metadata)
        self._input_model_yaml = input_model_yaml

    @classmethod
    def load(
        cls, path: PathType, device: t.Optional[str] = "cpu"
    ) -> torch.nn.Module:  # pylint: disable=arguments-differ
        """
        Load a detectron model from given yaml path.

        Args:
            path (`Union[str, os.PathLike]`, or :obj:`~bentoml._internal.types.PathType`):
                Given path containing saved yaml config for loading detectron model.
            device (`str`, `optional`, default to ``cpu``):
                Device type to cast model. Default behaviour similar to :obj:`torch.device("cuda")`
                Options: "cuda" or "cpu". If None is specified then return default config.MODEL.DEVICE

        Returns:
            :class:`torch.nn.Module`

        Raises:
            MissingDependencyException:
                ``detectron2`` is required by :class:`~bentoml.detectron.DetectronModel`.
        """  # noqa: E501

        cfg: "detectron2.config.CfgNode" = get_cfg()
        weight_path = cls.model_path(path, cls.PTH_FILE_EXTENSION)
        yaml_path = cls.model_path(path, cls.YAML_FILE_EXTENSION)

        cfg.merge_from_file(yaml_path)
        model: torch.nn.Module = build_model(cfg)
        model.eval()
        if device:
            model.to(device)

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(weight_path)
        return model

    def save(self, path: PathType) -> None:
        os.makedirs(path, exist_ok=True)
        checkpointer = DetectionCheckpointer(self._model, save_dir=path)
        checkpointer.save(self._MODEL_NAMESPACE)

        cfg: "detectron2.config.CfgNode" = get_cfg()
        if self._input_model_yaml:
            cfg.merge_from_file(self._input_model_yaml)

        with open(
            self.model_path(path, self.YAML_FILE_EXTENSION), 'w', encoding='utf-8'
        ) as ouf:
            ouf.write(cfg.dump())
