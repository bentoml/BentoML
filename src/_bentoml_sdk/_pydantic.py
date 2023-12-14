from __future__ import annotations

import typing as t

from pydantic._internal import _known_annotated_metadata
from typing_extensions import get_args
from typing_extensions import get_origin

from .types import DataframeSchema
from .types import PILImageEncoder
from .types import TensorSchema

if t.TYPE_CHECKING:
    from pydantic import ConfigDict

    Preparer: t.TypeAlias = t.Callable[
        [t.Any, t.Iterable[t.Any], ConfigDict], tuple[t.Any, list[t.Any]] | None
    ]
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import torch
    from PIL import Image
else:
    from bentoml._internal.utils.lazy_loader import LazyLoader

    np = LazyLoader("np", globals(), "numpy")
    tf = LazyLoader("tf", globals(), "tensorflow")
    torch = LazyLoader("torch", globals(), "torch")
    pd = LazyLoader("pd", globals(), "pandas")
    Image = LazyLoader("Image", globals(), "PIL.Image")


def numpy_prepare_pydantic_annotations(
    source: t.Any, annotations: t.Iterable[t.Any], config: ConfigDict
) -> tuple[t.Any, list[t.Any]] | None:
    if not getattr(source, "__module__", "").startswith("numpy"):
        return None

    origin = get_origin(source) or source

    if origin is not np.ndarray:
        return None

    args = get_args(source)
    dtype = np.dtype(args[1]).name if args else None

    _, remaining_annotations = _known_annotated_metadata.collect_known_metadata(
        annotations
    )
    schema = next(
        (a for a in remaining_annotations if isinstance(a, TensorSchema)), None
    )
    if schema is None:
        remaining_annotations.insert(0, TensorSchema("numpy-array", dtype=dtype))
    elif schema.dtype is None:
        schema.dtype = dtype
    return source, remaining_annotations


def torch_prepare_pydantic_annotations(
    source: t.Any, annotations: t.Iterable[t.Any], config: ConfigDict
) -> tuple[t.Any, list[t.Any]] | None:
    if not getattr(source, "__module__", "").startswith("torch"):
        return None

    origin = get_origin(source) or source

    if not issubclass(origin, torch.Tensor):
        return None

    _, remaining_annotations = _known_annotated_metadata.collect_known_metadata(
        annotations
    )
    if not any(isinstance(a, TensorSchema) for a in remaining_annotations):
        remaining_annotations.insert(0, TensorSchema("torch-tensor"))
    return origin, remaining_annotations


def tf_prepare_pydantic_annotations(
    source: t.Any, annotations: t.Iterable[t.Any], config: ConfigDict
) -> tuple[t.Any, list[t.Any]] | None:
    if not getattr(source, "__module__", "").startswith("tensorflow"):
        return None

    origin = get_origin(source) or source
    if not issubclass(origin, tf.Tensor):
        return None

    _, remaining_annotations = _known_annotated_metadata.collect_known_metadata(
        annotations
    )
    if not any(isinstance(a, TensorSchema) for a in remaining_annotations):
        remaining_annotations.insert(0, TensorSchema("tf-tensor"))
    return origin, remaining_annotations


def pandas_prepare_pydantic_annotations(
    source: t.Any, annotations: t.Iterable[t.Any], config: ConfigDict
) -> tuple[t.Any, list[t.Any]] | None:
    if not getattr(source, "__module__", "").startswith("pandas"):
        return None

    origin = get_origin(source) or source
    if not issubclass(origin, pd.DataFrame):
        return None

    _, remaining_annotations = _known_annotated_metadata.collect_known_metadata(
        annotations
    )
    if not any(isinstance(a, DataframeSchema) for a in remaining_annotations):
        remaining_annotations.insert(0, DataframeSchema())
    return origin, remaining_annotations


def pil_prepare_pydantic_annotations(
    source: t.Any, annotations: t.Iterable[t.Any], config: ConfigDict
) -> tuple[t.Any, list[t.Any]] | None:
    if not getattr(source, "__module__", "").startswith("PIL."):
        return None

    origin = get_origin(source) or source
    if not issubclass(origin, Image.Image):
        return None

    _, remaining_annotations = _known_annotated_metadata.collect_known_metadata(
        annotations
    )
    if not any(isinstance(a, PILImageEncoder) for a in remaining_annotations):
        remaining_annotations.insert(0, PILImageEncoder())
    return origin, remaining_annotations


def add_custom_preparers():
    try:
        from pydantic._internal import _std_types_schema
    except ModuleNotFoundError:
        return
    _std_types_schema.PREPARE_METHODS = (
        *_std_types_schema.PREPARE_METHODS,
        # tensors
        numpy_prepare_pydantic_annotations,
        torch_prepare_pydantic_annotations,
        tf_prepare_pydantic_annotations,
        # dataframe
        pandas_prepare_pydantic_annotations,
        # PIL image
        pil_prepare_pydantic_annotations,
    )
