import os
import typing as t
from pathlib import Path

from torch import nn

from bentoml._internal.artifacts import BaseArtifact
from bentoml._internal.exceptions import MissingDependencyException
from bentoml._internal.types import PathType


class DetectronModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`detectron2` models,
    in the form of :class:`~detectron2.checkpoint.DetectionCheckpointer`

    DetectronModel also contains the following attributes:

    - aug (`~detectron2.data.transforms.ResizeShortestEdge`):
        TODO:

    Args:
        model (`nn.Module`):
            detectron2 model of type :obj:`torch.nn.Module`
        input_model_yaml (`str`, `optional`):
            TODO:
        metadata (`Dict[str, Any]`, `optional``):
            Class metadata
        name (`str`, `optional`, default to `detectronmodel`):
            optional name for DetectronModel

    Raises:
        MissingDependencyException:
            :class:`~detectron2` is required by DetectronModel
        InvalidArgument:
            model is not an instance of :class:`torch.nn.Module`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    _model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        input_model_yaml: str = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        name: t.Optional[str] = "detectronmodel",
    ):
        super(DetectronModel, self).__init__(model, metadata=metadata, name=name)
        self._aug = None
        self._input_model_yaml = input_model_yaml

    @property
    def aug(self):
        return self._aug

    @classmethod
    def load(cls, path: PathType) -> nn.Module:
        try:
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.config import get_cfg
            from detectron2.data import transforms as T
            from detectron2.modeling import META_ARCH_REGISTRY

            if t.TYPE_CHECKING:
                from detectron2.config import CfgNode
        except ImportError:
            raise MissingDependencyException("detectron2 is required by DetectronModel")
        cfg: "CfgNode" = get_cfg()
        cfg.merge_from_file(str(cls.get_path(path, ".yaml")))
        meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
        model = meta_arch(cfg)
        model.eval()

        metadata = getattr(cls, "metadata")
        if metadata:
            device = metadata["device"]
            model.to(device)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(str(cls.get_path(path, ".pth")))
        cls.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        return model

    def save(self, path: PathType) -> None:
        try:
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.config import get_cfg

            if t.TYPE_CHECKING:
                from detectron2.config import CfgNode
        except ImportError:
            raise MissingDependencyException("detectron2 is required by DetectronModel")
        os.makedirs(path, exist_ok=True)
        checkpointer = DetectionCheckpointer(self._model, save_dir=path)
        checkpointer.save(self.__name__)
        cfg: "CfgNode" = get_cfg()
        cfg.merge_from_file(self._input_model_yaml)
        _path: Path = self.get_path(path, ".yaml")
        with _path.open('w', encoding='utf-8') as ouf:
            ouf.write(cfg.dump())
