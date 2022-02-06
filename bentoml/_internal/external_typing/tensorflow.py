import typing as t
from typing import ContextManager

from tensorflow.python.eager.context import LogicalDevice
from tensorflow.python.eager.context import PhysicalDevice
from tensorflow.python.eager.def_function import Function
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.eager.function import ConcreteFunction
from tensorflow.python.framework.dtypes import DType
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec
from tensorflow.python.framework.tensor_spec import TensorSpec as TFTensorSpec
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.training.tracking.base import Trackable
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorSpec
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.saved_model.function_deserialization import RestoredFunction

from bentoml._internal.external_typing.numpy import NpNDArray
from bentoml._internal.external_typing.numpy import NpDTypeLike


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

CastableTensorType = t.Union[Tensor, EagerTensor, SparseTensor, IndexedSlices]

TensorType = t.Union[Tensor, EagerTensor, SparseTensor, RaggedTensor, TensorArray]
TensorSpec = t.Union[
    TFTensorSpec,
    RaggedTensorSpec,
    SparseTensorSpec,
    TensorArraySpec,
]
def cast(x: t.Union[Tensor, SparseTensor, IndexedSlices], dtype: DType, name: t.Optional[str] = None) -> t.Union[
    Tensor, SparseTensor, IndexedSlices]:
    ...

class SignatureMap(t.Mapping[str, t.Any], Trackable):
    """A collection of SavedModel signatures."""
    _signatures: t.Dict[str, t.Any]

    def __init__(self) -> None:...

    # Ideally this object would be immutable, but restore is streaming so we do
    # need a private API for adding new signatures to an existing object.
    def _add_signature(self, name: str, concrete_function: ConcreteFunction) -> None:
        """Adds a signature to the _SignatureMap."""

    def __getitem__(self, key: str) -> t.Any: ...

    def __iter__(self) -> t.Iterator[str]:...

    def __len__(self) -> int:...

    def __repr__(self) -> str:...

    def _list_functions_for_serialization(self, unused_serialization_cache) -> t.Dict[str, t.Union[ConcreteFunction, Function]]:...


__all__ = [
    "CastableTensorType",
    "TensorType",
    "TensorSpec",
    "cast",
    "Trackable",
    "AutoTrackable",
    "ConcreteFunction",
    "RestoredFunction",
    "SignatureMap"
]
