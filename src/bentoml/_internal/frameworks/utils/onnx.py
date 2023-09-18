from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

from ...types import LazyType
from ....exceptions import BentoMLException
from ...utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import onnx
    import torch

    from ... import external_typing as ext
    from ...external_typing import tensorflow as tf_ext

    ONNXArgTensorType = (
        ext.NpNDArray
        | ext.PdDataFrame
        | torch.Tensor
        | tf_ext.Tensor
        | list[int | float | str]
    )
    ONNXArgSequenceType = list["ONNXArgType"]
    ONNXArgMapKeyType = int | str
    ONNXArgMapType = dict[ONNXArgMapKeyType, "ONNXArgType"]
    ONNXArgType = ONNXArgMapType | ONNXArgTensorType | ONNXArgSequenceType

    ONNXArgCastedType = (
        ext.NpNDArray
        | list["ONNXArgCastedType"]
        | dict[ONNXArgMapKeyType, "ONNXArgCastedType"]
    )
    ONNXArgCastingFuncType = t.Callable[[ONNXArgType], ONNXArgCastedType]
    ONNXArgCastingFuncGeneratorType = t.Callable[
        [dict[str, t.Any]], t.Callable[[ONNXArgType], ONNXArgCastedType]
    ]

else:
    np = LazyLoader("np", globals(), "numpy")
    onnx = LazyLoader(
        "onnx",
        globals(),
        "onnx",
        exc_msg="`onnx` is required to use bentoml.onnx module.",
    )

logger = logging.getLogger(__name__)

TENSORPROTO_ELEMENT_TYPE_TO_NUMPY_TYPE: dict[int, str] = {
    onnx.TensorProto.FLOAT: "float32",  # 1
    onnx.TensorProto.UINT8: "uint8",  # 2
    onnx.TensorProto.INT8: "int8",  # 3
    onnx.TensorProto.UINT16: "uint16",  # 4
    onnx.TensorProto.INT16: "int16",  # 5
    onnx.TensorProto.INT32: "int32",  # 6
    onnx.TensorProto.INT64: "int64",  # 7
    onnx.TensorProto.STRING: "str",  # 8 or "unicode"?
    onnx.TensorProto.BOOL: "bool",  # 9
    onnx.TensorProto.FLOAT16: "float16",  # 10
    onnx.TensorProto.DOUBLE: "double",  # 11
    onnx.TensorProto.UINT32: "uint32",  # 12
    onnx.TensorProto.UINT64: "uint64",  # 13
    onnx.TensorProto.COMPLEX64: "csingle",  # 14
    onnx.TensorProto.COMPLEX128: "cdouble",  # 15
    # onnx.TensorProto.BFLOAT16: None,  # 16
}

# type -> casting function generator
CASTING_FUNC_DISPATCHER: dict[str, ONNXArgCastingFuncGeneratorType] = {}


def gen_input_casting_func(spec: dict[str, t.Any]) -> ONNXArgCastingFuncType:
    return _gen_input_casting_func(spec["type"])


def _gen_input_casting_func(sig: dict[str, t.Any]) -> ONNXArgCastingFuncType:
    input_types = list(sig.keys())
    if len(input_types) != 1:
        raise BentoMLException(
            "onnx model input type dictionary should have only one key!"
        )
    input_type = input_types[0]
    input_spec = sig[input_type]
    return CASTING_FUNC_DISPATCHER[input_type](input_spec)


def _gen_input_casting_func_for_tensor(
    sig: dict[str, t.Any]
) -> t.Callable[[ONNXArgTensorType], ext.NpNDArray]:
    elem_type = sig["elemType"]
    to_dtype = TENSORPROTO_ELEMENT_TYPE_TO_NUMPY_TYPE[elem_type]

    def _mapping(item: ONNXArgTensorType) -> ext.NpNDArray:
        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
            item = item.astype(to_dtype, copy=False)
        elif isinstance(item, list):
            item = np.array(item).astype(to_dtype, copy=False)
        elif LazyType["ext.PdDataFrame"]("pandas.DataFrame").isinstance(item):
            item = item.to_numpy(dtype=to_dtype)
        elif LazyType["tf.Tensor"]("tensorflow.Tensor").isinstance(item):
            item = np.array(memoryview(item)).astype(to_dtype, copy=False)  # type: ignore
        elif LazyType["torch.Tensor"]("torch.Tensor").isinstance(item):
            item = item.numpy().astype(to_dtype, copy=False)
        else:
            raise TypeError(
                "`run` of ONNXRunnable only takes `numpy.ndarray`, `pd.DataFrame`, `tf.Tensor`, `torch.Tensor` or a list as input Tensor type"
            )
        return t.cast("ext.NpNDArray", item)

    return _mapping


CASTING_FUNC_DISPATCHER["tensorType"] = t.cast(
    "ONNXArgCastingFuncGeneratorType", _gen_input_casting_func_for_tensor
)


def _gen_input_casting_func_for_map(
    sig: dict[str, t.Any]
) -> t.Callable[[ONNXArgMapType], dict[ONNXArgMapKeyType, ONNXArgCastedType]]:
    map_value_sig = t.cast(dict[str, t.Any], sig["valueType"])
    value_casting_func = _gen_input_casting_func(map_value_sig)

    def _mapping(item: ONNXArgMapType) -> dict[ONNXArgMapKeyType, t.Any]:
        new_item = {k: value_casting_func(v) for k, v in item.items()}
        return new_item

    return _mapping


CASTING_FUNC_DISPATCHER["mapType"] = t.cast(
    "ONNXArgCastingFuncGeneratorType", _gen_input_casting_func_for_map
)


def _gen_input_casting_func_for_sequence(
    sig: dict[str, t.Any]
) -> t.Callable[[ONNXArgSequenceType], list[t.Any]]:
    seq_elem_sig = t.cast(dict[str, t.Any], sig["elemType"])
    elem_casting_func = _gen_input_casting_func(seq_elem_sig)

    def _mapping(item: ONNXArgSequenceType) -> list[t.Any]:
        new_item = list(elem_casting_func(elem) for elem in item)
        return new_item

    return _mapping


CASTING_FUNC_DISPATCHER["sequenceType"] = t.cast(
    "ONNXArgCastingFuncGeneratorType", _gen_input_casting_func_for_sequence
)
