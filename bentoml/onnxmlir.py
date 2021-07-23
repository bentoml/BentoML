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
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

MT = t.TypeVar("MT")

try:
    # this has to be able to find the arch and OS specific PyRuntime .so file
    from PyRuntime import ExecutionSession
except ImportError:
    raise MissingDependencyException("PyRuntime package library must be in python path")


class OnnxMlirModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`onnx-mlir` compiled models.
    onnx-mlir is a compiler technology that can take an onnx model and lower it
    (using llvm) to an inference library that is optimized and has little external
    dependencies.

    The PyRuntime interface is created during the build of onnx-mlir using pybind.
    See the onnx-mlir supporting documentation for detail.

    Args:
        model (`str`):
            Given filepath or protobuf of converted model. Make sure to use corresponding library
            to convert model from different frameworks to ONNX format
        backend (`str`, `optional`, default to `onnxruntime`):
            Name of ONNX inference runtime. ["onnxruntime", "onnxruntime-gpu"]
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`PyRuntime` must be accessible in path.

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    ONNXMLIR_MODEL_EXTENSION: str = ".so"

    def __init__(self, model: str, metadata: t.Optional[MetadataType] = None):
        super(OnnxMlirModel, self).__init__(model, metadata=metadata)
        self._inference_session = None
        self._model_so_path = None

    @classmethod
    def load(cls, path: PathType):  # type: ignore
        model_path: str = cls.get_path(path, cls.ONNXMLIR_MODEL_EXTENSION)
        inference_session = ExecutionSession(model_path, "run_main_graph")
        return inference_session

    def save(self, path: PathType) -> None:
        # copies the model .so and places in the version controlled deployment path
        shutil.copyfile(self._model, self.get_path(path, self.ONNXMLIR_MODEL_EXTENSION))
