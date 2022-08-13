"""
Static mapping from BentoML protobuf message to values.

For all function in this module, make sure to lazy load the generated protobuf.
"""
from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING
from functools import lru_cache

from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    from enum import Enum

    import grpc
    import numpy as np

    from bentoml.grpc.v1 import service_pb2 as pb

    from ... import external_typing as ext
else:
    grpc = LazyLoader(
        "grpc",
        globals(),
        "grpc",
        exc_msg="'grpc' is required. Install with 'pip install grpcio'.",
    )
    pb = LazyLoader("pb", globals(), "bentoml.grpc.v1.service_pb2")
    np = LazyLoader("np", globals(), "numpy")

# Maps HTTP status code to grpc.StatusCode
@lru_cache(maxsize=1)
def http_status_to_grpc_status_map() -> dict[Enum, grpc.StatusCode]:
    return {
        HTTPStatus.OK: grpc.StatusCode.OK,
        HTTPStatus.UNAUTHORIZED: grpc.StatusCode.UNAUTHENTICATED,
        HTTPStatus.FORBIDDEN: grpc.StatusCode.PERMISSION_DENIED,
        HTTPStatus.NOT_FOUND: grpc.StatusCode.UNIMPLEMENTED,
        HTTPStatus.TOO_MANY_REQUESTS: grpc.StatusCode.UNAVAILABLE,
        HTTPStatus.BAD_GATEWAY: grpc.StatusCode.UNAVAILABLE,
        HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
        HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
        HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
        HTTPStatus.UNPROCESSABLE_ENTITY: grpc.StatusCode.FAILED_PRECONDITION,
    }


@lru_cache(maxsize=1)
def grpc_status_to_http_status_map() -> dict[grpc.StatusCode, Enum]:
    return {v: k for k, v in http_status_to_grpc_status_map().items()}


@lru_cache(maxsize=1)
def filetype_pb_to_mimetype_map() -> dict[pb.File.FileType.ValueType, str]:
    return {
        pb.File.FILE_TYPE_CSV: "text/csv",
        pb.File.FILE_TYPE_PLAINTEXT: "text/plain",
        pb.File.FILE_TYPE_JSON: "application/json",
        pb.File.FILE_TYPE_BYTES: "application/octet-stream",
        pb.File.FILE_TYPE_PDF: "application/pdf",
        pb.File.FILE_TYPE_PNG: "image/png",
        pb.File.FILE_TYPE_JPEG: "image/jpeg",
        pb.File.FILE_TYPE_GIF: "image/gif",
        pb.File.FILE_TYPE_TIFF: "image/tiff",
        pb.File.FILE_TYPE_BMP: "image/bmp",
        pb.File.FILE_TYPE_WEBP: "image/webp",
        pb.File.FILE_TYPE_SVG: "image/svg+xml",
    }


@lru_cache(maxsize=1)
def mimetype_to_filetype_pb_map() -> dict[str, pb.File.FileType.ValueType]:
    return {v: k for k, v in filetype_pb_to_mimetype_map().items()}


# TODO: support the following types for for protobuf message:
# - support complex64, complex128, object and struct types
# - BFLOAT16, QINT32, QINT16, QUINT16, QINT8, QUINT8
#
# For int16, uint16, int8, uint8 -> specify types in NumpyNdarray + using int_values.
#
# For bfloat16, half (float16) -> specify types in NumpyNdarray + using float_values.
#
# for string_values, use <U for np.dtype instead of S (zero-terminated bytes).
FIELDPB_TO_NPDTYPE_NAME_MAP = {
    "bool_values": "bool",
    "float_values": "float32",
    "string_values": "<U",
    "double_values": "float64",
    "int32_values": "int32",
    "int64_values": "int64",
    "uint32_values": "uint32",
    "uint64_values": "uint64",
}


@lru_cache(maxsize=1)
def dtypepb_to_fieldpb_map() -> dict[pb.NDArray.DType.ValueType, str]:
    return {
        pb.NDArray.DTYPE_FLOAT: "float_values",
        pb.NDArray.DTYPE_DOUBLE: "double_values",
        pb.NDArray.DTYPE_INT32: "int32_values",
        pb.NDArray.DTYPE_INT64: "int64_values",
        pb.NDArray.DTYPE_UINT32: "uint32_values",
        pb.NDArray.DTYPE_UINT64: "uint64_values",
        pb.NDArray.DTYPE_BOOL: "bool_values",
        pb.NDArray.DTYPE_STRING: "string_values",
    }


@lru_cache(maxsize=1)
def fieldpb_to_dtypepb_map() -> dict[str, pb.NDArray.DType.ValueType]:
    return {v: k for k, v in dtypepb_to_fieldpb_map().items()}


@lru_cache(maxsize=1)
def dtypepb_to_npdtype_map() -> dict[pb.NDArray.DType.ValueType, ext.NpDTypeLike]:
    # pb.NDArray.Dtype -> np.dtype
    return {
        k: np.dtype(FIELDPB_TO_NPDTYPE_NAME_MAP[v])
        for k, v in dtypepb_to_fieldpb_map().items()
    }


@lru_cache(maxsize=1)
def npdtype_to_dtypepb_map() -> dict[ext.NpDTypeLike, pb.NDArray.DType.ValueType]:
    # np.dtype -> pb.NDArray.Dtype
    return {v: k for k, v in dtypepb_to_npdtype_map().items()}


@lru_cache(maxsize=1)
def fieldpb_to_npdtype_map() -> dict[str, ext.NpDTypeLike]:
    # str -> np.dtype
    return {k: np.dtype(v) for k, v in FIELDPB_TO_NPDTYPE_NAME_MAP.items()}


@lru_cache(maxsize=1)
def npdtype_to_fieldpb_map() -> dict[ext.NpDTypeLike, str]:
    # np.dtype -> str
    return {v: k for k, v in fieldpb_to_npdtype_map().items()}
