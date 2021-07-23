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

import shutil
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import BentoMLException, MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import onnx
    import onnxruntime
except ImportError:
    raise MissingDependencyException('"onnx" package is required by OnnxModel')


class OnnxModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`onnx` models.

    Args:
        model (`str`):
            Given filepath or protobuf of converted model.
            Make sure to use corresponding library to convert
            model from different frameworks to ONNX format.
        backend (`str`, `optional`, default to `onnxruntime`):
            Name of ONNX inference runtime. ["onnxruntime", "onnxruntime-gpu"]
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata.

    Raises:
        MissingDependencyException:
            :obj:`onnx` is required by OnnxModel
        NotImplementedError:
            :obj:`backend` as onnx runtime is not supported by ONNX
        BentoMLException:
            :obj:`backend` as onnx runtime is not supported by OnnxModel
        InvalidArgument:
            :obj:`path` passed in :meth:`~save` is not either
             a :obj:`onnx.ModelProto` or filepath

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    SUPPORTED_ONNX_BACKEND: t.List[str] = ["onnxruntime", "onnxruntime-gpu"]
    ONNX_EXTENSION: str = ".onnx"

    def __init__(
        self,
        model: str,
        backend: t.Optional[str] = "onnxruntime",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(OnnxModel, self).__init__(model, metadata=metadata)
        if backend not in self.SUPPORTED_ONNX_BACKEND:
            raise BentoMLException(
                f'"{backend}" runtime is currently not supported for OnnxModel'
            )
        self._backend = backend

    @classmethod
    def __model_file__path(cls, path: PathType) -> PathType:
        return cls.get_path(path, cls.ONNX_EXTENSION)

    @classmethod
    def load(
        cls, path: t.Union[PathType, onnx.ModelProto]
    ) -> "onnxruntime.InferenceSession":
        if isinstance(path, onnx.ModelProto):
            return onnxruntime.InferenceSession(path.SerializeToString())
        else:
            _get_path: str = cls.__model_file__path(path)
            return onnxruntime.InferenceSession(_get_path)

    def save(self, path: t.Union[PathType, "onnx.ModelProto"]) -> None:
        if isinstance(self._model, onnx.ModelProto):
            onnx.save_model(self._model, self.__model_file__path(path))
        else:
            shutil.copyfile(self._model, self.__model_file__path(path))
