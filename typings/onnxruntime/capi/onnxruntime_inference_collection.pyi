from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from onnx.onnx_pb import ModelProto
from onnxruntime.capi import _pybind_state as C
from torch import ctypes

def get_ort_device_type(device: str) -> OrtDevice: ...
def check_and_normalize_provider_args(
    providers: Optional[Union[str, Dict[str, Any]]],
    provider_options: Optional[Dict[str, Any]],
    available_provider_names: List[str],
) -> Tuple[Union[str, Dict[str, Any]]]: ...

class Session:
    def __init__(self) -> None: ...
    def get_session_options(self) -> C.SessionOptions: ...
    def get_inputs(self) -> List[C.NodeArg]: ...
    def get_outputs(self) -> List[C.NodeArg]: ...
    def get_overridable_initializers(self) -> List[C.NodeArg]: ...
    def get_modelmeta(self) -> C.ModelMetadata: ...
    def get_providers(self) -> List[str]: ...
    def get_provider_options(self) -> List[Tuple[str, Dict[str, Any]]]: ...
    def set_providers(
        self,
        providers: Optional[Tuple[str, Dict[str, Any]]] = ...,
        provider_options: Optional[Dict[str, Any]] = ...,
    ) -> None: ...
    def disable_fallback(self) -> None: ...
    def enable_fallback(self) -> None: ...
    def run(
        self,
        output_names: List[str],
        input_feed: Dict[str, Any],
        run_options: C.RunOptions = ...,
    ) -> None: ...
    def run_with_ort_values(
        self,
        output_names: List[str],
        input_dict_ort_values: Dict[str, OrtValue],
        run_options: C.RunOptions = ...,
    ) -> None: ...
    def end_profiling(self) -> None: ...
    def get_profiling_start_time_ns(self) -> float: ...
    def io_binding(self) -> IOBinding: ...
    def run_with_iobinding(
        self, iobinding: IOBinding, run_options: C.RunOptions = ...
    ) -> None: ...

class InferenceSession(Session):
    def __init__(
        self,
        path_or_bytes: Union[str, os.PathLike[str], BytesIO, ModelProto],
        sess_options: Optional[C.SessionOptions] = ...,
        providers: Optional[Union[str, List[Union[str, Tuple[str, Dict[str, Any]]]]]] = ...,
        provider_options: Optional[Dict[str, Any]] = ...,
        **kwargs: str
    ) -> None: ...

class IOBinding:
    def __init__(self, session: InferenceSession) -> None: ...
    def bind_cpu_input(
        self, name: str, arr_on_cpu: Union[List[int], "np.ndarray[Any, np.dtype[Any]]"]
    ): ...
    def bind_input(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: Union[str, type, OrtDevice],
        shape: Tuple[int, ...],
        buffer_ptr: ctypes.c_char_p,
    ) -> None: ...
    def bind_ortvalue_input(self, name: str, ortvalue: OrtValue): ...
    def bind_output(
        self,
        name: str,
        device_type: str = ...,
        device_id: int = ...,
        element_type: Union[str, type, OrtDevice] = ...,
        shape: Tuple[int, ...] = ...,
        buffer_ptr: ctypes.c_char_p = ...,
    ): ...
    def bind_ortvalue_output(self, name: str, ortvalue: OrtValue): ...
    def get_outputs(self) -> List[OrtValue]: ...
    def copy_outputs_to_cpu(self) -> None: ...
    def clear_binding_inputs(self) -> None: ...
    def clear_binding_outputs(self) -> None: ...

class OrtMemoryInfo:
    name: str
    mem_type: C.OrtMemType = ...
    alloc_type: C.OrtAllocatorType = ...
    device: OrtDevice = ...

class OrtValue:
    def __init__(
        self, ortvalue: OrtValue, numpy_obj: "np.ndarray[Any, np.dtype[Any]]" = ...
    ) -> None: ...
    @staticmethod
    def ortvalue_from_numpy(
        numpy_obj: "np.ndarray[Any, np.dtype[Any]]",
        device_type: Union[str, type, OrtDevice] = ...,
        device_id: int = ...,
    ) -> OrtValue: ...
    @staticmethod
    def ortvalue_from_shape_and_type(
        shape: List[Union[int, Tuple[int, ...]]] = ...,
        element_type: "np.dtype[Any]" = ...,
        device_type: Union[str, type, OrtDevice] = ...,
        device_id: int = ...,
    ) -> OrtValue: ...
    @staticmethod
    def ort_value_from_sparse_tensor(sparse_tensor: SparseTensor) -> OrtValue: ...
    def as_sparse_tensor(self) -> SparseTensor: ...
    def data_ptr(self) -> ctypes.c_char_p: ...
    def device_name(self) -> str: ...
    def shape(self) -> Tuple[int, ...]: ...
    def data_type(self) -> str: ...
    def is_tensor(self) -> bool: ...
    def is_sparse_tensor(self) -> bool: ...
    def is_tensor_sequence(self) -> bool: ...
    def numpy(self) -> "np.ndarray[Any, np.dtype[Any]]": ...

class OrtDevice:
    CPU: Literal[0] = ...
    GPU: Literal[1] = ...
    FPGA: Literal[2] = ...
    def __init__(self, c_ort_device: OrtDevice) -> None: ...
    @staticmethod
    def make(ort_device_name: str, device_id: int): ...
    def device_id(self) -> int: ...
    def device_type(self) -> str: ...

class SparseTensor:
    def __init__(self, sparse_tensor: SparseTensor) -> None: ...
    @staticmethod
    def sparse_coo_from_numpy(
        dense_shape: "np.ndarray[Any, np.dtype[np.int64]]",
        values: "np.ndarray[Any, np.dtype[np.int64]]",
        coo_indices: "np.ndarray[Any, np.dtype[np.int64]]",
        ort_device: OrtDevice,
    ) -> SparseTensor: ...
    @staticmethod
    def sparse_csr_from_numpy(
        dense_shape: "np.ndarray[Any, np.dtype[np.int64]]",
        values: "np.ndarray[Any, np.dtype[np.int64]]",
        inner_indices: "np.ndarray[Any, np.dtype[np.int64]]",
        outer_indices: "np.ndarray[Any, np.dtype[np.int64]]",
        ort_device: OrtDevice,
    ) -> SparseTensor: ...
    def values(self) -> "np.ndarray[Any, np.dtype[Any]]": ...
    def as_coo_view(self) -> "np.ndarray[Any, np.dtype[Any]]": ...
    def as_csrc_view(self) -> "np.ndarray[Any, np.dtype[Any]]": ...
    def as_blocksparse_view(self) -> "np.ndarray[Any, np.dtype[Any]]": ...
    def to_cuda(self, ort_device: OrtDevice) -> SparseTensor: ...
    def format(self) -> Any: ...
    def dense_shape(self) -> Tuple[int, ...]: ...
    def data_type(self) -> str: ...
    def device_name(self) -> str: ...
