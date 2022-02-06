# type: ignore[reportMissingTypeStubs]
import os
import typing as t
import pathlib
from types import ModuleType
from typing import ContextManager

from tensorflow_hub import Module as HubModule
from tensorflow_hub import KerasLayer
from tensorflow.keras.models import Model as KerasModel
from tensorflow.python.eager.context import LogicalDevice
from tensorflow.python.eager.context import PhysicalDevice
from tensorflow.python.eager.context import _EagerDeviceContext
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.module.module import Module
from tensorflow.python.client.session import Session as TFSession
from tensorflow.python.eager.function import FunctionSpec
from tensorflow.python.eager.function import ConcreteFunction
from tensorflow.python.framework.dtypes import DType
from tensorflow.core.protobuf.config_pb2 import ConfigProto as TFConfigProto
from tensorflow.python.eager.def_function import Function
from tensorflow.python.framework.type_spec import TypeSpec
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec as TFTensorArraySpec
from tensorflow.python.framework.tensor_spec import TensorSpec as TFTensorSpec
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.training.tracking.base import Trackable
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import (
    SparseTensorSpec as TFSparseTensorSpec,
)
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import (
    RaggedTensorSpec as TFRaggedTensorSpec,
)
from tensorflow.python.saved_model.save_options import SaveOptions as TFSaveOptions
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.saved_model.function_deserialization import RestoredFunction

from .numpy import NpNDArray
from .numpy import NpDTypeLike


# NOTE: Tensorflow core team is considering to remove this in the future
# added types here for compatibility with Tensorflow V1
class EagerTensor(Tensor):
    """Base class for EagerTensor."""

    def __complex__(self) -> Tensor:
        ...

    def __int__(self) -> Tensor:
        ...

    def __long__(self) -> Tensor:
        ...

    def __float__(self) -> Tensor:
        ...

    def __index__(self) -> Tensor:
        ...

    def __bool__(self) -> t.NoReturn:
        ...

    __nonzero__ = __bool__

    def __format__(self, format_spec: str) -> Tensor:
        ...

    def __reduce__(
        self,
    ) -> t.Tuple[t.Callable[..., Tensor], t.Tuple[NpNDArray, ...]]:
        ...

    def __copy__(self) -> Tensor:
        ...

    def __deepcopy__(self, memo: t.Any) -> Tensor:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __len__(self) -> t.NoReturn:
        ...

    def __array__(self, dtype: t.Optional[NpDTypeLike] = None) -> NpNDArray:
        ...

    def _numpy_internal(self) -> NpNDArray:
        ...

    def _numpy(self) -> NpNDArray:
        ...

    @property
    def dtype(self) -> DType:
        ...

    def numpy(self) -> NpNDArray:
        ...

    @property
    def backing_device(self) -> t.Union[LogicalDevice, PhysicalDevice]:
        ...

    def _datatype_enum(self) -> t.NoReturn:
        ...

    def _shape_tuple(self) -> t.Tuple[int, ...]:
        ...

    def _rank(self) -> int:
        ...

    def _num_elements(self) -> int:
        ...

    def _copy_to_device(self, device_name: str) -> Tensor:
        ...

    @staticmethod
    def _override_operator(name: str, func: t.Callable[..., t.Any]) -> None:
        ...

    def _copy_nograd(
        self,
        ctx: t.Optional[ContextManager] = None,
        device_name: t.Optional[str] = None,
    ) -> Tensor:
        ...

    def _copy(
        self,
        ctx: t.Optional[ContextManager] = None,
        device_name: t.Optional[str] = None,
    ) -> Tensor:
        ...

    @property
    def shape(self) -> TensorShape:
        ...

    def get_shape(self) -> TensorShape:
        ...

    def _shape_as_list(self) -> t.Optional[t.List[t.Optional[int]]]:
        ...

    @property
    def ndim(self) -> int:
        ...

    def set_shape(self, shape: t.Union[t.Tuple[int, ...], t.List[int]]):
        ...


class RaggedTensorSpec(TFRaggedTensorSpec):
    @classmethod
    def from_value(cls, value: RaggedTensor) -> TFRaggedTensorSpec:
        ...


class SparseTensorSpec(TFSparseTensorSpec):
    @classmethod
    def from_value(cls, value: t.Union[SparseTensor, NpNDArray]) -> TFSparseTensorSpec:
        ...


class TensorArraySpec(TFTensorArraySpec):
    @staticmethod
    def from_value(value: TensorArray) -> TFTensorArraySpec:
        ...


class TensorSpec(TFTensorSpec):
    @classmethod
    def from_tensor(cls, value: Tensor, name: t.Optional[str] = None) -> TFTensorSpec:
        ...


class SignatureMap(t.Mapping[str, t.Any], Trackable):
    """A collection of SavedModel signatures."""

    _signatures: t.Dict[str, t.Union[ConcreteFunction, RestoredFunction, Function]]

    def __init__(self) -> None:
        ...

    # Ideally this object would be immutable, but restore is streaming so we do
    # need a private API for adding new signatures to an existing object.
    def _add_signature(self, name: str, concrete_function: ConcreteFunction) -> None:
        """Adds a signature to the _SignatureMap."""

    def __getitem__(
        self, key: str
    ) -> t.Union[ConcreteFunction, RestoredFunction, Function]:
        ...

    def __iter__(self) -> t.Iterator[str]:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...


class test(ModuleType):
    @staticmethod
    def is_gpu_available() -> bool:
        ...


class config(ModuleType):
    @staticmethod
    def list_physical_devices(
        device_type: t.Optional[str],
    ) -> t.List[t.Union[LogicalDevice, PhysicalDevice]]:
        ...


class compat(ModuleType):
    class v1(ModuleType):
        @staticmethod
        def global_variables_initializer() -> t.Any:
            ...

        @staticmethod
        def get_default_graph() -> t.Any:
            ...

        class Session(TFSession):
            ...

        class saved_model(ModuleType):
            def load_v2(
                export_dir: t.Union[str, os.PathLike[str], bytes, pathlib.Path],
                tags: t.Optional[t.List[str]] = None,
                options: t.Optional[TFSaveOptions] = None,
            ) -> AutoTrackable:
                ...

        class ConfigProto(TFConfigProto):
            ...

    class v2(ModuleType):
        class saved_model(ModuleType):
            def load(
                export_dir: t.Union[str, os.PathLike[str], bytes, pathlib.Path],
                tags: t.Optional[t.List[str]],
                options: t.Optional[TFSaveOptions],
            ) -> AutoTrackable:
                ...

            def save(
                obj: AutoTrackable,
                export_dir: t.Union[str, os.PathLike[str], bytes, pathlib.Path],
                signatures: t.Optional[ConcreteFunction],
                options: t.Optional[TFSaveOptions],
            ) -> t.NoReturn:
                ...

            class SaveOptions(TFSaveOptions):
                ...


class saved_model(ModuleType):  # tf2
    def load(
        export_dir: t.Union[str, os.PathLike[str], bytes, pathlib.Path],
        tags: t.Optional[t.List[str]] = None,
        options: t.Optional[TFSaveOptions] = None,
    ) -> AutoTrackable:
        ...

    def save(
        obj: AutoTrackable,
        export_dir: t.Union[str, os.PathLike[str], bytes, pathlib.Path],
        signatures: t.Optional[ConcreteFunction] = None,
        options: t.Optional[TFSaveOptions] = None,
    ) -> t.NoReturn:
        ...

    class SaveOptions(TFSaveOptions):
        ...


def cast(
    x: t.Union[Tensor, SparseTensor, IndexedSlices],
    dtype: DType,
    name: t.Optional[str] = None,
) -> t.Union[Tensor, SparseTensor, IndexedSlices]:
    ...


def constant(
    value: t.Union[t.List[t.Any]],
    dtype: t.Optional[DType] = None,
    shape: t.Optional[t.Union[t.Tuple[int], t.List[int]]] = None,
    name: t.Optional[str] = "Const",
) -> Tensor:
    ...


def device(device_id: str) -> _EagerDeviceContext:
    ...


def convert_to_tensor(
    value: t.Union[Tensor, NpNDArray, t.List[t.Union[int, float]]],
    dtype: t.Optional[DType],
    name: t.Optional[str],
) -> Tensor:
    ...


CastableTensorType = t.Union[Tensor, EagerTensor, SparseTensor, IndexedSlices]
TensorType = t.Union[Tensor, EagerTensor, SparseTensor, RaggedTensor, TensorArray]
UnionTensorSpec = t.Union[
    TensorSpec,
    RaggedTensorSpec,
    SparseTensorSpec,
    TensorArraySpec,
]

__all__ = [
    "CastableTensorType",
    "TensorType",
    "UnionTensorSpec",
    "cast",
    "constant",
    "device",
    "convert_to_tensor",
    "Trackable",
    "AutoTrackable",
    "ConcreteFunction",
    "RestoredFunction",
    "FunctionSpec",
    "SignatureMap",
    "Module",
    "KerasLayer",
    "HubModule",
    "KerasModel",
    "TypeSpec",
    "test",
    "config",
    "saved_model",
    "compat",
]
