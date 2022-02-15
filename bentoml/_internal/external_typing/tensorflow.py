import typing as t

from tensorflow.python.framework.ops import Tensor
from tensorflow.python.module.module import Module
from tensorflow.python.client.session import Session
from tensorflow.python.eager.function import FunctionSpec
from tensorflow.python.eager.def_function import Function
from tensorflow.python.framework.type_spec import TypeSpec
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.training.tracking.base import Trackable
from tensorflow.python.framework.sparse_tensor import SparseTensorSpec
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.saved_model.save_options import SaveOptions
from tensorflow.python.framework.composite_tensor import CompositeTensor
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.saved_model.function_deserialization import RestoredFunction

try:
    from tensorflow.python.types.core import GenericFunction
    from tensorflow.python.types.core import ConcreteFunction
    from tensorflow.python.framework.ops import _EagerTensorBase as EagerTensor
except ImportError:
    from tensorflow.python.eager.function import ConcreteFunction

    class GenericFunction(t.Protocol):
        def __call__(self, *args: t.Any, **kwargs: t.Any):
            ...

    # NOTE: future proof when tensorflow decided to remove EagerTensor
    # and enable eager execution on Tensor by default. This class only serves
    # as type fallback.
    class EagerTensor(Tensor):
        ...


class SignatureMap(t.Mapping[str, t.Any], Trackable):
    """A collection of SavedModel signatures."""

    _signatures: t.Dict[str, t.Union[ConcreteFunction, RestoredFunction, Function]]

    def __init__(self) -> None:
        ...

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


# This denotes all decorated function with `tf.function`
DecoratedFunction = t.Union[RestoredFunction, ConcreteFunction, GenericFunction]

# This denotes all possible tensor type that can be casted with `tf.cast`
CastableTensorType = t.Union[Tensor, EagerTensor, CompositeTensor, IndexedSlices]
# This encapsulates scenarios where input can be accepted as TensorArray
TensorLike = t.Union[CastableTensorType, TensorArray]
# This defines all possible TensorSpec from tensorflow API
UnionTensorSpec = t.Union[
    TensorSpec,
    RaggedTensorSpec,
    SparseTensorSpec,
    TensorArraySpec,
]

# TODO(aarnphm): Specify types instead of t.Any
TensorSignature = t.Tuple[TensorSpec, bool, t.Optional[t.Any]]
InputSignature = t.Tuple[TensorSignature, t.Dict[str, TypeSpec]]

# This denotes all Keras Model API
KerasModel = t.Union[Model, Sequential]

__all__ = [
    "CastableTensorType",
    "TensorLike",
    "InputSignature",
    "TensorSignature",
    "UnionTensorSpec",
    "Trackable",
    "AutoTrackable",
    "ConcreteFunction",
    "RestoredFunction",
    "FunctionSpec",
    "SignatureMap",
    "Module",
    "TypeSpec",
    "SaveOptions",
    "Session",
    "KerasModel",
]
