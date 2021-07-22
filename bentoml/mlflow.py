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
from types import ModuleType

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException
from ._internal.types import MetadataType, PathType

MT = t.TypeVar("MT")


class MLflowModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`mlflow` models

    Args:
        model (`mlflow.models.Model`):
            All mlflow models are of type :obj:`mlflow.models.Model`
        loader_module (`types.ModuleType`):
            flavors supported by :obj:`mlflow`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`mlflow` is required by MLflowModel
        ArtifactLoadingException:
            given `loader_module` is not supported by :obj:`mlflow`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    try:
        import mlflow
    except ImportError:
        raise MissingDependencyException("mlflow is required by MLflowModel")

    def __init__(
        self,
        model: MT,
        loader_module: ModuleType,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(MLflowModel, self).__init__(model, metadata=metadata)
        if 'mlflow' not in loader_module.__name__:
            raise InvalidArgument('given `loader_module` is not omitted by mlflow.')
        self.__load__module(loader_module)

    @classmethod
    def __load__module(cls, module: ModuleType):
        setattr(cls, "_loader_module", module)
        return cls._loader_module

    @classmethod
    def load(cls, path: PathType) -> MT:
        try:
            import mlflow
        except ImportError:
            raise MissingDependencyException("mlflow is required by MLflowModel")
        project_path: str = str(os.path.join(path, cls._MODEL_NAMESPACE))
        return mlflow.pyfunc.load_model(project_path)

    def save(self, path: PathType) -> None:
        self._loader_module.save_model(self._model, self.get_path(path))
