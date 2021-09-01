import os
import shutil
import typing as t

import bentoml._internal.constants as const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader, _flatten_list, catch_exceptions
from .exceptions import BentoMLException, MissingDependencyException

_exc = const.IMPORT_ERROR_MSG.format(
    fwr="onnxruntime & onnx",
    module=__name__,
    inst="Refers to https://onnxruntime.ai/"
    " to correctly install backends options"
    " and platform suitable for your application usecase.",
)

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import onnx
    import onnxruntime
else:
    onnx = LazyLoader("onnx", globals(), "onnx")
    onnxruntime = LazyLoader("onnxruntime", globals(), "onnxruntime")


class ONNXModel(Model):
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

    One then can define :code:`bento.py`::

        TODO:
    """

    SUPPORTED_ONNX_BACKEND: t.List[str] = ["onnxruntime", "onnxruntime-gpu"]
    ONNX_EXTENSION: str = ".onnx"

    def __init__(
        self,
        model: t.Union[PathType, "onnx.ModelProto"],
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
        return os.path.join(path, f"{MODEL_NAMESPACE}{cls.ONNX_EXTENSION}")

    @classmethod
    @catch_exceptions(
        catch_exc=ModuleNotFoundError, throw_exc=MissingDependencyException, msg=_exc
    )
    def load(  # pylint: disable=arguments-differ
        cls,
        path: t.Union[PathType, "onnx.ModelProto"],
        backend: t.Optional[str] = "onnxruntime",
        providers: t.List[t.Union[str, t.Tuple[str, dict]]] = None,
        sess_opts: t.Optional["onnxruntime.SessionOptions"] = None,
    ) -> "onnxruntime.InferenceSession":
        if backend not in cls.SUPPORTED_ONNX_BACKEND:
            raise BentoMLException(
                f'"{backend}" runtime is currently not supported for ONNXModel'
            )
        if providers is not None:
            if not all(
                i in onnxruntime.get_all_providers() for i in _flatten_list(providers)
            ):
                raise BentoMLException(
                    f"'{providers}' can't be parsed by `onnxruntime`"
                )
        else:
            providers = onnxruntime.get_available_providers()
        if isinstance(path, onnx.ModelProto):
            return onnxruntime.InferenceSession(
                path.SerializeToString(), sess_options=sess_opts, providers=providers
            )
        else:
            _get_path = str(cls.__get_model_fpath(path))
            return onnxruntime.InferenceSession(
                _get_path, sess_options=sess_opts, providers=providers
            )

    @catch_exceptions(
        catch_exc=ModuleNotFoundError, throw_exc=MissingDependencyException, msg=_exc
    )
    def save(self, path: t.Union[PathType, "onnx.ModelProto"]) -> None:
        if isinstance(self._model, onnx.ModelProto):
            onnx.save_model(self._model, self.__get_model_fpath(path))
        else:
            shutil.copyfile(self._model, str(self.__get_model_fpath(path)))
