# ruff: noqa: E402

from __future__ import annotations

from bentoml._internal.utils import warn_deprecated

warn_deprecated(
    "`bentoml.io` is deprecated since BentoML v1.4 and will be removed in a future version. Please upgrade to new style IO types instead.",
    stacklevel=1,
)

from ._internal.io_descriptors import from_spec
from ._internal.io_descriptors.base import IODescriptor
from ._internal.io_descriptors.file import File
from ._internal.io_descriptors.image import Image
from ._internal.io_descriptors.json import JSON
from ._internal.io_descriptors.multipart import Multipart
from ._internal.io_descriptors.numpy import NumpyNdarray
from ._internal.io_descriptors.pandas import PandasDataFrame
from ._internal.io_descriptors.pandas import PandasSeries
from ._internal.io_descriptors.text import Text
from ._internal.io_descriptors.utils import SSE

__all__ = [
    "File",
    "Image",
    "IODescriptor",
    "JSON",
    "Multipart",
    "NumpyNdarray",
    "PandasDataFrame",
    "PandasSeries",
    "Text",
    "from_spec",
    "SSE",
]
