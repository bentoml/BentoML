"""
Static mapping from BentoML protobuf message to values.

For all function in this module, make sure to lazy load the generated protobuf.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from functools import lru_cache

from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    from bentoml.grpc.v1 import service_pb2 as pb
else:
    pb = LazyLoader("pb", globals(), "bentoml.grpc.v1.service_pb2")


@lru_cache(maxsize=1)
def file_enum_mapping() -> dict[pb.File.FileType.ValueType, str]:
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


def mimetype_from_file_enum(pb_type: pb.File.FileType.ValueType) -> str:
    return file_enum_mapping()[pb_type]


def file_enum_from_mimetype(mime_type: str) -> pb.File.FileType.ValueType:
    return {v: k for k, v in file_enum_mapping().items()}[mime_type]
