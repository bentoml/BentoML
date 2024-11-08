from __future__ import annotations

import contextlib
import fnmatch
import functools
import io
import operator
import os
import tempfile
import typing as t
from pathlib import Path
from pathlib import PurePath

import attrs
from annotated_types import BaseMetadata
from pydantic_core import core_schema
from starlette.datastructures import UploadFile

from bentoml._internal.utils import dict_filter_none

from .typing_utils import is_file_like
from .typing_utils import is_image_type

if t.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import torch
    from pydantic import GetCoreSchemaHandler
    from pydantic import GetJsonSchemaHandler
    from typing_extensions import Literal

    TensorType = t.Union[np.ndarray[t.Any, t.Any], tf.Tensor, torch.Tensor]
    TensorFormat = Literal["numpy-array", "tf-tensor", "torch-tensor"]
    from PIL import Image as PILImage
else:
    from bentoml._internal.utils.lazy_loader import LazyLoader

    np = LazyLoader("np", globals(), "numpy")
    tf = LazyLoader("tf", globals(), "tensorflow")
    torch = LazyLoader("torch", globals(), "torch")
    pa = LazyLoader("pa", globals(), "pyarrow")
    pd = LazyLoader("pd", globals(), "pandas")
    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")

T = t.TypeVar("T")

# This is an internal global state that is True when the model is being serialized for arrow
__in_arrow_serialization__ = False


@contextlib.contextmanager
def arrow_serialization():
    global __in_arrow_serialization__
    __in_arrow_serialization__ = True
    try:
        yield
    finally:
        __in_arrow_serialization__ = False


class PILImageEncoder:
    def decode(self, obj: bytes | t.BinaryIO | UploadFile | PILImage.Image) -> t.Any:
        if is_image_type(type(obj)):
            return t.cast("PILImage.Image", obj)
        if isinstance(obj, UploadFile):
            formats = None
            if obj.headers.get("Content-Type", "").startswith("image/"):
                formats = [obj.headers["Content-Type"][6:].upper()]
            return PILImage.open(obj.file, formats=formats)
        if is_file_like(obj):
            return PILImage.open(obj)
        if isinstance(obj, bytes):
            return PILImage.open(io.BytesIO(obj))
        return obj

    def encode(self, obj: PILImage.Image) -> bytes:
        buffer = io.BytesIO()
        obj.save(buffer, format=obj.format or "PNG")
        return buffer.getvalue()

    def __get_pydantic_core_schema__(
        self, source: type[t.Any], handler: t.Callable[[t.Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            function=self.decode,
            schema=core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(self.encode),
        )

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, t.Any]:
        value = handler(schema)
        if handler.mode == "validation":
            value.update({"type": "file", "format": "image", "pil": True})
        else:
            value.update({"type": "string", "format": "binary"})
        return value


@attrs.define
class FileSchema:
    format: str = "binary"
    content_type: str | None = None

    def __attrs_post_init__(self) -> None:
        if self.content_type is not None:
            self.format = self.content_type.split("/")[0]

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, t.Any]:
        value = handler(schema)
        if handler.mode == "validation":
            value.update({"type": "file", "format": self.format})
            if self.content_type is not None:
                value.update({"content_type": self.content_type})
        else:
            value.update({"type": "string", "format": "binary"})
        return value

    def encode(self, obj: Path) -> bytes:
        return obj.read_bytes()

    def decode(self, obj: bytes | t.BinaryIO | UploadFile | PurePath | str) -> t.Any:
        from bentoml._internal.context import request_temp_dir

        media_type: str | None = None

        if isinstance(obj, str):
            return obj
        if isinstance(obj, PurePath):
            return Path(obj)
        if isinstance(obj, UploadFile):
            body = obj.file.read()
            filename = obj.filename
            media_type = obj.content_type
        elif is_file_like(obj):
            body = obj.read()
            filename = (
                os.path.basename(fn)
                if (fn := getattr(obj, "name", None)) is not None
                else None
            )
        elif isinstance(obj, bytes):
            body = obj
            filename = None
        else:
            from pydantic_core import PydanticCustomError

            raise PydanticCustomError("path_type", "Invalid file type")
        if media_type is not None and self.content_type is not None:
            if not fnmatch.fnmatch(media_type, self.content_type):
                raise ValueError(
                    f"Invalid content type {media_type}, expected {self.content_type}"
                )
        with tempfile.NamedTemporaryFile(
            suffix=filename, dir=request_temp_dir(), delete=False
        ) as f:
            f.write(body)
            return Path(f.name)

    def __get_pydantic_core_schema__(
        self, source: type[t.Any], handler: t.Callable[[t.Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            function=self.decode,
            schema=core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(self.encode),
        )


@attrs.frozen(unsafe_hash=True)
class TensorSchema:
    format: TensorFormat
    dtype: t.Optional[str] = None
    shape: t.Optional[t.Tuple[int, ...]] = None

    @property
    def dim(self) -> int | None:
        if self.shape is None:
            return None
        return functools.reduce(operator.mul, self.shape, 1)

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, t.Any]:
        value = handler(schema)
        if handler.mode == "validation":
            value.update(
                dict_filter_none(
                    {
                        "type": "tensor",
                        "format": self.format,
                        "dtype": self.dtype,
                        "shape": self.shape,
                        "dim": self.dim,
                    }
                )
            )
        else:
            dimension = 1 if self.shape is None else len(self.shape)
            child = {"type": "number"}
            for _ in range(dimension):
                child = {"type": "array", "items": child}
            value.update(child)
        return value

    def __get_pydantic_core_schema__(
        self, source_type: t.Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            self.validate,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.encode, info_arg=True
            ),
        )

    def encode(self, arr: TensorType, info: core_schema.SerializationInfo) -> t.Any:
        if self.format == "numpy-array":
            assert isinstance(arr, np.ndarray)
            numpy_array = arr
        elif self.format == "tf-tensor":
            if not info.mode_is_json():  # tf.Tensor supports picklev5 serialization
                return arr
            numpy_array = arr.numpy()
        else:
            assert isinstance(arr, torch.Tensor)
            if arr.device.type != "cpu":
                numpy_array = arr.cpu().numpy()
            else:
                numpy_array = arr.numpy()
        if __in_arrow_serialization__:
            numpy_array = numpy_array.flatten()
        if info.mode_is_json():
            return numpy_array.tolist()
        return numpy_array

    @property
    def framework_dtype(self) -> t.Any:
        dtype = self.dtype
        if dtype is None:
            return None
        if self.format == "numpy-array":
            return getattr(np, dtype)
        elif self.format == "tf-tensor":
            return getattr(tf, dtype)
        else:
            return getattr(torch, dtype)

    def validate(self, obj: t.Any) -> t.Any:
        arr: t.Any
        if self.format == "numpy-array":
            if isinstance(obj, np.ndarray):
                return obj
            arr = np.array(obj, dtype=self.framework_dtype)
            if self.shape is not None:
                arr = arr.reshape(self.shape)
            return arr
        elif self.format == "tf-tensor":
            if isinstance(obj, tf.Tensor):
                return obj
            else:
                return tf.constant(obj, dtype=self.framework_dtype, shape=self.shape)  # type: ignore
        else:
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, np.ndarray):
                return torch.from_numpy(obj)
            arr = torch.tensor(obj, dtype=self.framework_dtype)
            if self.shape is not None:
                arr = arr.reshape(self.shape)
            return arr


@attrs.frozen(unsafe_hash=True)
class DataframeSchema:
    orient: str = "records"
    columns: tuple[str] | None = attrs.field(
        default=None,
        converter=lambda x: tuple(x) if x else None,
    )

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, t.Any]:
        value = handler(schema)
        if handler.mode == "validation":
            value.update(
                dict_filter_none(
                    {
                        "type": "dataframe",
                        "orient": self.orient,
                        "columns": self.columns,
                    }
                )
            )
        else:
            if self.orient == "records":
                value.update(
                    {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                )
            elif self.orient == "columns":
                value.update(
                    {
                        "type": "object",
                        "additionalProperties": {"type": "array"},
                    }
                )
            else:
                raise ValueError(
                    "Only 'records' and 'columns' are supported for orient"
                )
        return value

    def __get_pydantic_core_schema__(
        self, source_type: t.Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            self.validate,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                self.encode, info_arg=True
            ),
        )

    def encode(self, df: pd.DataFrame, info: core_schema.SerializationInfo) -> t.Any:
        if not info.mode_is_json():
            return df
        if self.orient == "records":
            return df.to_dict(orient="records")
        elif self.orient == "columns":
            return df.to_dict(orient="list")
        else:
            raise ValueError("Only 'records' and 'columns' are supported for orient")

    def validate(self, obj: t.Any) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj
        return pd.DataFrame(obj, columns=self.columns)


@attrs.frozen
class ContentType(BaseMetadata):
    content_type: str


@attrs.frozen
class Shape(BaseMetadata):
    dimensions: tuple[int, ...]


@attrs.frozen
class DType(BaseMetadata):
    dtype: str
