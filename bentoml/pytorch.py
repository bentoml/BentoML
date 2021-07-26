import logging
import os
import typing as t
import zipfile

import cloudpickle
import torch.nn

from ._internal.models.base import MODEL_NAMESPACE, PT_EXTENSION, Model
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import torch
except ImportError:
    raise MissingDependencyException(
        "torch is required by PyTorchModel and PyTorchLightningModel"
    )
try:
    import pytorch_lightning
except ImportError:
    pytorch_lightning = None

logger = logging.getLogger(__name__)


class PyTorchModel(Model):
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

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(
        self,
        model: t.Union[torch.nn.Module, torch.jit.ScriptModule],
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PyTorchModel, self).__init__(model, metadata=metadata)

    @staticmethod
    def __get_weight_fpath(path: PathType) -> PathType:
        return os.path.join(path, f"{MODEL_NAMESPACE}{PT_EXTENSION}")

    @classmethod
    def load(
        cls, path: PathType
    ) -> t.Union["torch.nn.Module", "torch.jit.ScriptModule"]:
        # TorchScript Models are saved as zip files
        if zipfile.is_zipfile(cls.__get_weight_fpath(path)):
            return torch.jit.load(cls.__get_weight_fpath(path))
        else:
            return cloudpickle.load(open(cls.__get_weight_fpath(path), "rb"))

    def save(self, path: PathType) -> None:
        # If model is a TorchScriptModule, we cannot apply standard pickling
        if isinstance(self._model, torch.jit.ScriptModule):
            torch.jit.save(self._model, self.__get_weight_fpath(path))
            return

        cloudpickle.dump(self._model, open(self.__get_weight_fpath(path), "wb"))


class PyTorchLightningModel(Model):
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

    One then can define :code:`bento.py`::

        TODO:

    """

    def __init__(
        self,
        model: pytorch_lightning.LightningModule,
        metadata: t.Optional[MetadataType] = None,
    ):  # noqa
        if pytorch_lightning is None:
            raise MissingDependencyException(
                "pytorch_lightning is required by PyTorchLightningModel"
            )
        super(PyTorchLightningModel, self).__init__(model, metadata=metadata)

    @staticmethod
    def __get_weight_fpath(path: PathType) -> str:
        return str(os.path.join(path, f"{MODEL_NAMESPACE}{PT_EXTENSION}"))

    @classmethod
    def load(cls, path: PathType) -> "torch.jit.ScriptModule":
        return torch.jit.load(cls.__get_weight_fpath(path))

    def save(self, path: PathType) -> None:
        torch.jit.save(self._model.to_torchscript(), self.__get_weight_fpath(path))
