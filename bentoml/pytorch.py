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

import logging
import typing as t
import zipfile

import torch.nn

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException
from ._internal.types import MetadataType, PathType
from ._internal.utils import cloudpickle

try:
    import torch
except ImportError:
    raise MissingDependencyException(
        "torch is required by PyTorchModel and PyTorchLightningModel"
    )

logger = logging.getLogger(__name__)


class PyTorchModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`pytorch` models.

    Args:
        model (`Union[torch.nn.Module, torch.jit.ScriptModule]`):
            every PyTorch model is of instance :obj:`torch.nn.Module`. :class:`PyTorchModel`
            additionally accept :obj:`torch.jit.ScriptModule` parsed as :obj:`model` argument.
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

    def __init__(
        self,
        model: t.Union[torch.nn.Module, torch.jit.ScriptModule],
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PyTorchModel, self).__init__(model, metadata=metadata)

    @classmethod
    def __get_weight_file__path(cls, path: PathType) -> PathType:
        return cls.get_path(path, cls.PT_EXTENSION)

    @classmethod
    def load(cls, path: PathType) -> t.Union[torch.nn.Module, torch.jit.ScriptModule]:
        # TorchScript Models are saved as zip files
        if zipfile.is_zipfile(cls.__get_weight_file__path(path)):
            model: torch.jit.ScriptModule = torch.jit.load(
                cls.__get_weight_file__path(path)
            )
        else:
            model: torch.nn.Module = cloudpickle.load(
                open(cls.__get_weight_file__path(path), 'rb')
            )
        return model

    def save(self, path: PathType) -> None:
        # If model is a TorchScriptModule, we cannot apply standard pickling
        if isinstance(self._model, torch.jit.ScriptModule):
            return torch.jit.save(self._model, self.__get_weight_file__path(path))

        return cloudpickle.dump(
            self._model, open(self.__get_weight_file__path(path), 'wb')
        )


class PyTorchLightningModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`pytorch_lightning` models.

    Args:
        model (`Union[torch.nn.Module, torch.jit.ScriptModule]`):
            every PyTorch model is of instance :obj:`torch.nn.Module`. :class:`PyTorchModel`
            additionally accept :obj:`torch.jit.ScriptModule` parsed as :obj:`model` argument.
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`torch` and :obj:`pytorch_lightning` is required by PyTorchLightningModel
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
        import pytorch_lightning as pl
        import pytorch_lightning.core
    except ImportError:
        raise MissingDependencyException(
            "pytorch_lightning is required by PyTorchLightningModel"
        )

    _model: pl.core.LightningModule

    def __init__(
        self,
        model: pytorch_lightning.core.LightningModule,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PyTorchLightningModel, self).__init__(model, metadata=metadata)

    @classmethod
    def __get_weight_file__path(cls, path: PathType) -> PathType:
        return cls.get_path(path, cls.PT_EXTENSION)

    @classmethod
    def load(cls, path: PathType) -> pl.core.LightningModule:
        return torch.jit.load(cls.__get_weight_file__path(path))

    def save(self, path: PathType) -> None:
        torch.jit.save(self._model.to_torchscript(), self.__get_weight_file__path(path))