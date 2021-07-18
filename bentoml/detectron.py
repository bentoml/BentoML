import os
import typing as t

from torch import nn

from bentoml._internal.artifacts import BaseArtifact
from bentoml._internal.exceptions import MissingDependencyException
from bentoml._internal.types import PathType

try:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import META_ARCH_REGISTRY

    if t.TYPE_CHECKING:
        from detectron2.config import CfgNode
except ImportError:
    raise MissingDependencyException("detectron2 is required by DetectronModel")


class DetectronModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`detectron2` models,
    in the form of :class:`~detectron2.checkpoint.DetectionCheckpointer`

    DetectronModel also contains the following attributes:

    - aug (`~detectron2.data.transforms.ResizeShortestEdge`):
        TODO:

    Args:
        model (`torch.nn.Module`):
            detectron2 model is of type :obj:`torch.nn.Module`
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
    PYTORCH_FILE_EXTENSION: str = ".pth"
    YAML_FILE_EXTENSION: str = ".yaml"

    def __init__(
        self,
        model: nn.Module,
        input_model_yaml: str = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        name: t.Optional[str] = "detectronmodel",
    ):
        super(DetectronModel, self).__init__(model, metadata=metadata, name=name)
        self._input_model_yaml = input_model_yaml

    @classmethod
    def load(cls, path: PathType, device: t.Optional[str] = "cpu") -> nn.Module:
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

        cfg: "CfgNode" = get_cfg()
        weight_path = str(cls.get_path(path, cls.PYTORCH_FILE_EXTENSION))
        yaml_path = str(cls.get_path(path, cls.YAML_FILE_EXTENSION))

        cfg.merge_from_file(yaml_path)
        # ugh why is this creating a different model from the same meta_architecture
        model: nn.Module = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg)
        model.eval()
        if device:
            model.to(device)
        else:
            model.to(cfg.MODEL.DEVICE)

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(weight_path)
        return model

    def save(self, path: PathType) -> None:
        os.makedirs(path, exist_ok=True)
        checkpointer = DetectionCheckpointer(self._model, save_dir=path)
        checkpointer.save(self.name)

        cfg: "CfgNode" = get_cfg()
        if self._input_model_yaml:
            cfg.merge_from_file(self._input_model_yaml)

        with open(
            self.model_path(path, self.YAML_FILE_EXTENSION), 'w', encoding='utf-8'
        ) as ouf:
            ouf.write(cfg.dump())
