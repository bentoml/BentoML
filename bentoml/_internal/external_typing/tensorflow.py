import typing as t

from tensorflow.python.framework.ops import Tensor
from tensorflow.python.module.module import Module
from tensorflow.python.eager.function import FunctionSpec
from tensorflow.python.eager.function import ConcreteFunction
from tensorflow.python.eager.def_function import Function
from tensorflow.python.framework.type_spec import TypeSpec
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.training.tracking.base import Trackable
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorSpec
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.saved_model.function_deserialization import RestoredFunction
from tensorflow.python.saved_model.save_options import SaveOptions


class EagerTensor(Tensor): ...


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
    "Trackable",
    "AutoTrackable",
    "ConcreteFunction",
    "RestoredFunction",
    "FunctionSpec",
    "SignatureMap",
    "Module",
    "TypeSpec",
    "SaveOptions"
]
