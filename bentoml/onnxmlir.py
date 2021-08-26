import os
import shutil
import typing as t

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

MT = t.TypeVar("MT")

if t.TYPE_CHECKING:
    from PyRuntime import ExecutionSession  # pylint: disable=unused-import
else:
    ExecutionSession = LazyLoader(
        "ExecutionSession", globals(), "PyRuntime.ExecutionSession"
    )


class ONNXMlirModel(Model):
    """
    Model class for saving/loading :obj:`onnx-mlir` compiled models.
    onnx-mlir is a compiler technology that can take an onnx model and lower it
    (using llvm) to an inference library that is optimized and has little external
    dependencies.

    The PyRuntime interface is created during the build of onnx-mlir using pybind.
    See the onnx-mlir supporting documentation for detail.

    Args:
        model (`str`):
            Given filepath or protobuf of converted model.
             Make sure to use corresponding library to convert
             model from different frameworks to ONNX format
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`PyRuntime` must be accessible in path.

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    ONNXMLIR_EXTENSION: str = ".so"

    def __init__(self, model: str, metadata: t.Optional[MetadataType] = None):
        super(ONNXMlirModel, self).__init__(model, metadata=metadata)
        self._inference_session = None
        self._model_so_path = None

    @classmethod
    def load(cls, path: PathType) -> "ExecutionSession":
        model_path: str = os.path.join(
            path, f"{MODEL_NAMESPACE}{cls.ONNXMLIR_EXTENSION}"
        )
        inference_session = ExecutionSession(model_path, "run_main_graph")
        return inference_session

    def save(self, path: PathType) -> None:
        # copies the model .so and places in the version controlled deployment path
        shutil.copyfile(
            self._model,
            os.path.join(path, f"{MODEL_NAMESPACE}{self.ONNXMLIR_EXTENSION}"),
        )
