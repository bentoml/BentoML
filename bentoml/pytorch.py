import logging
import typing as t
import zipfile

import cloudpickle
import torch.nn

from ._internal.artifacts.base import MT, ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

logger = logging.getLogger(__name__)


class PyTorchModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`pytorch` models.

    Args:
        model (`Union[torch.nn.Module, torch.jit.ScriptModule]`):
            Accepts either `torch.nn.Module` or
             `torch.jit.ScriptModule`.
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`torch` is required by PyTorchModel
        InvalidArgument:
            :obj:`model` is not an instance of :class:`torch.nn.Module`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    try:
        import torch
    except ImportError:
        raise MissingDependencyException(
            "torch is required by PyTorchModel and PyTorchLightningModel"
        )

    def __init__(
        self,
        model: t.Union[torch.nn.Module, torch.jit.ScriptModule],
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PyTorchModel, self).__init__(model, metadata=metadata)

    @classmethod
    def __get_weight_fpath(cls, path: PathType) -> PathType:
        return cls.get_path(path, cls.PT_EXTENSION)

    @classmethod
    def load(cls, path: PathType) -> t.Union[torch.nn.Module, torch.jit.ScriptModule]:
        # TorchScript Models are saved as zip files
        if zipfile.is_zipfile(cls.__get_weight_fpath(path)):
            model: torch.jit.ScriptModule = torch.jit.load(cls.__get_weight_fpath(path))
        else:
            model: torch.nn.Module = cloudpickle.load(
                open(cls.__get_weight_fpath(path), "rb")
            )
        return model

    def save(self, path: PathType) -> None:
        # If model is a TorchScriptModule, we cannot apply standard pickling
        if isinstance(self._model, torch.jit.ScriptModule):
            return torch.jit.save(self._model, self.__get_weight_fpath(path))

        return cloudpickle.dump(self._model, open(self.__get_weight_fpath(path), "wb"))


class PyTorchLightningModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`pytorch_lightning` models.

    Args:
        model (`pytorch_lightning.LightningModule`):
            Accepts `pytorch_lightning.LightningModule`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`torch` and :obj:`pytorch_lightning` is required
             by PyTorchLightningModel
        InvalidArgument:
            :obj:`model` is not an instance of
             :class:`pytorch_lightning.LightningModule`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self, model, metadata: t.Optional[MetadataType] = None,
    ):

        try:
            import pytorch_lightning as pl  # noqa # pylint: disable=unused-import
            import pytorch_lightning.core  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "pytorch_lightning is required by PyTorchLightningModel"
            )
        super(PyTorchLightningModel, self).__init__(model, metadata=metadata)

    @classmethod
    def __get_weight_fpath(cls, path: PathType) -> PathType:
        return cls.get_path(path, cls.PT_EXTENSION)

    @classmethod
    def load(cls, path: PathType) -> MT:
        return torch.jit.load(cls.__get_weight_fpath(path))

    def save(self, path: PathType) -> None:
        torch.jit.save(self._model.to_torchscript(), self.__get_weight_fpath(path))
