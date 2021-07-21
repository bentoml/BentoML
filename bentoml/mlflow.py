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

import importlib
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import ArtifactLoadingException, MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import mlflow
except ImportError:
    raise MissingDependencyException('mlflow is required by MLflowModel')

MT = t.TypeVar("MT")


class MLflowModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`mlflow` models

    Args:
        model (`mlflow.models.Model`):
            All mlflow models are of type :obj:`mlflow.models.Model`
        loader_module (`str`):
            flavors supported by :obj:`mlflow`
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
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
    """  # noqa: E501

    def __init__(
        self, model: MT, loader_module: str, metadata: t.Optional[MetadataType] = None
    ):
        super(MLflowModel, self).__init__(model, metadata=metadata)
        __module = (
            loader_module if 'mlflow.' in loader_module else f'mlflow.{loader_module}'
        )
        self.__module__loader(__module)

    @classmethod
    def __module_name__path(cls, path: PathType) -> PathType:
        return cls.model_path(path, f"_module_name{cls.TXT_FILE_EXTENSION}")

    @classmethod
    def __module__loader(cls, module: str):
        try:
            __loader_module = importlib.import_module(module)
        except ImportError:
            raise ArtifactLoadingException(
                f"Failed to import mlflow.{module}. Make sure that the given flavor is supported by MLflow"
            )
        setattr(cls, '_loader_module', __loader_module)
        return cls._loader_module

    # fmt: off
    @classmethod
    def load(cls, path: PathType) -> MT:  # noqa # pylint: disable=arguments-differ
        with open(cls.__module_name__path(path), 'rb') as txt_file:
            module = txt_file.read().decode(cls._FILE_ENCODING)
            cls.__module__loader(module)

        # walk_path will actually returns the first occurrence of given path. With MLflow we
        # only care about the directory of given model structure, thus getting the parents.
        model_path: str = str(cls.walk_path(path, "").parents[0])
        return cls._loader_module.load_model(model_path)
    # fmt: on

    def save(self, path: PathType) -> None:
        with open(self.__module_name__path(path), 'wb') as txt_file:
            txt_file.write(self._loader_module.__name__.encode(self._FILE_ENCODING))
        self._loader_module.save_model(self._model, self.model_path(path, ""))
