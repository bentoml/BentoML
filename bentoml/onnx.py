import shutil
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import BentoMLException, MissingDependencyException

try:
    import onnx
    import onnxruntime
except ImportError:
    raise MissingDependencyException('"onnx" package is required by ONNXModel')


class ONNXModel(ModelArtifact):
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
            :obj:`onnx` is required by ONNXModel
        NotImplementedError:
            :obj:`backend` as onnx runtime is not supported by ONNX
        BentoMLException:
            :obj:`backend` as onnx runtime is not supported by ONNXModel
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
        super(ONNXModel, self).__init__(model, metadata=metadata)
        if backend not in self.SUPPORTED_ONNX_BACKEND:
            raise BentoMLException(
                f'"{backend}" runtime is currently not supported for ONNXModel'
            )
        self._backend = backend

    @classmethod
    def __get_model_fpath(cls, path: PathType) -> PathType:
        return cls.get_path(path, cls.ONNX_EXTENSION)

    @classmethod
    def load(
        cls, path: t.Union[PathType, onnx.ModelProto]
    ) -> "onnxruntime.InferenceSession":
        if isinstance(path, onnx.ModelProto):
            return onnxruntime.InferenceSession(path.SerializeToString())
        else:
            _get_path: str = cls.__get_model_fpath(path)
            return onnxruntime.InferenceSession(_get_path)

    def save(self, path: t.Union[PathType, "onnx.ModelProto"]) -> None:
        if isinstance(self._model, onnx.ModelProto):
            onnx.save_model(self._model, self.__get_model_fpath(path))
        else:
            shutil.copyfile(self._model, self.__get_model_fpath(path))
