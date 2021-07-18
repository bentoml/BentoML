import os
import typing as t

from torch import nn

from bentoml._internal.artifacts import BaseArtifact
from bentoml._internal.exceptions import MissingDependencyException


class DetectronModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`detectron2` models,
    in the form of :class:`~detectron2.checkpoint.DetectionCheckpointer`

    Args:

    Raises:
        MissingDependencyException:
            :class:`~detectron2` is required by DetectronModel
        InvalidArgument:
            model is not an instance of :class:`torch.nn.Module`

    Example usage under :code:`train.py`::

        TODO

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self,
        model: nn.Module,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        name: t.Optional[str] = "detectronmodel",
    ):
        super(DetectronModel, self).__init__(model, metadata=metadata, name=name)
        self._aug = None
        self._input_model_yaml = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    @classmethod
    def load(cls, path):
        try:
            from detectron2.checkpoint import (  # noqa # pylint: disable=unused-import
                DetectionCheckpointer,
            )
            from detectron2.config import get_cfg
            from detectron2.data import transforms as T
            from detectron2.modeling import META_ARCH_REGISTRY
        except ImportError:
            raise MissingDependencyException("detectron2 is required by DetectronModel")
        cfg = get_cfg()
        cfg.merge_from_file(f"{path}/{self.name}.yaml")
        meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
        self._model = meta_arch(cfg)
        self._model.eval()

        device = self._metadata["device"]
        self._model.to(device)
        checkpointer = DetectionCheckpointer(self._model)
        checkpointer.load(f"{path}/{self.name}.pth")
        self._aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        return self.pack(self._model)

    def save(self, dst):
        try:
            from detectron2.checkpoint import (  # noqa # pylint: disable=unused-import
                DetectionCheckpointer,
            )
            from detectron2.config import get_cfg
        except ImportError:
            raise MissingDependencyException(
                "detectron2 package is required to use DetectronArtifact"
            )
        os.makedirs(dst, exist_ok=True)
        checkpointer = DetectionCheckpointer(self._model, save_dir=dst)
        checkpointer.save(self.name)
        cfg = get_cfg()
        cfg.merge_from_file(self._input_model_yaml)
        with open(
            os.path.join(dst, f"{self.name}.yaml"), "w", encoding="utf-8"
        ) as output_file:
            output_file.write(cfg.dump())
