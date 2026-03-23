from __future__ import annotations

import collections
import datetime
import io
import pathlib
import pickle
import typing as t
import uuid
from decimal import Decimal

PickleGlobal = tuple[str, str]

_SAFE_BUILTIN_GLOBALS: dict[PickleGlobal, t.Any] = {
    ("builtins", "bytearray"): bytearray,
    ("builtins", "complex"): complex,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "range"): range,
    ("builtins", "set"): set,
    ("builtins", "slice"): slice,
    ("collections", "Counter"): collections.Counter,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("collections", "defaultdict"): collections.defaultdict,
    ("collections", "deque"): collections.deque,
    ("datetime", "date"): datetime.date,
    ("datetime", "datetime"): datetime.datetime,
    ("datetime", "time"): datetime.time,
    ("datetime", "timedelta"): datetime.timedelta,
    ("datetime", "timezone"): datetime.timezone,
    ("decimal", "Decimal"): Decimal,
    ("pathlib", "Path"): pathlib.Path,
    ("pathlib", "PosixPath"): pathlib.PosixPath,
    ("pathlib", "PurePath"): pathlib.PurePath,
    ("pathlib", "PurePosixPath"): pathlib.PurePosixPath,
    ("pathlib", "PureWindowsPath"): pathlib.PureWindowsPath,
    ("pathlib", "WindowsPath"): pathlib.WindowsPath,
    ("uuid", "UUID"): uuid.UUID,
}
_OPTIONAL_SAFE_GLOBAL_FACTORIES: dict[PickleGlobal, t.Callable[[], t.Any]] = {
    ("numpy", "dtype"): lambda: __import__("numpy").dtype,
    ("numpy", "ndarray"): lambda: __import__("numpy").ndarray,
    ("numpy._core.multiarray", "scalar"): lambda: (
        __import__("numpy._core.multiarray", fromlist=["scalar"]).scalar
    ),
    ("numpy._core.multiarray", "_reconstruct"): lambda: (
        __import__("numpy._core.multiarray", fromlist=["_reconstruct"])._reconstruct
    ),
    ("numpy._core.numeric", "_frombuffer"): lambda: (
        __import__("numpy._core.numeric", fromlist=["_frombuffer"])._frombuffer
    ),
    ("numpy.core.multiarray", "scalar"): lambda: (
        __import__("numpy.core.multiarray", fromlist=["scalar"]).scalar
    ),
    ("numpy.core.multiarray", "_reconstruct"): lambda: (
        __import__("numpy.core.multiarray", fromlist=["_reconstruct"])._reconstruct
    ),
    ("numpy.core.numeric", "_frombuffer"): lambda: (
        __import__("numpy.core.numeric", fromlist=["_frombuffer"])._frombuffer
    ),
    ("pandas._libs.internals", "_unpickle_block"): lambda: (
        __import__(
            "pandas._libs.internals", fromlist=["_unpickle_block"]
        )._unpickle_block
    ),
    ("pandas.core.frame", "DataFrame"): lambda: (
        __import__("pandas.core.frame", fromlist=["DataFrame"]).DataFrame
    ),
    ("pandas.core.indexes.base", "Index"): lambda: (
        __import__("pandas.core.indexes.base", fromlist=["Index"]).Index
    ),
    ("pandas.core.indexes.base", "_new_Index"): lambda: (
        __import__("pandas.core.indexes.base", fromlist=["_new_Index"])._new_Index
    ),
    ("pandas.core.indexes.range", "RangeIndex"): lambda: (
        __import__("pandas.core.indexes.range", fromlist=["RangeIndex"]).RangeIndex
    ),
    ("pandas.core.internals.managers", "BlockManager"): lambda: (
        __import__(
            "pandas.core.internals.managers", fromlist=["BlockManager"]
        ).BlockManager
    ),
    ("PIL.Image", "Image"): lambda: __import__("PIL.Image", fromlist=["Image"]).Image,
    ("tensorflow.python.framework.ops", "convert_to_tensor"): lambda: (
        __import__(
            "tensorflow.python.framework.ops", fromlist=["convert_to_tensor"]
        ).convert_to_tensor
    ),
    ("torch._utils", "_rebuild_tensor_v2"): lambda: (
        __import__("torch._utils", fromlist=["_rebuild_tensor_v2"])._rebuild_tensor_v2
    ),
    ("torch.storage", "_load_from_bytes"): lambda: _safe_torch_storage_load_from_bytes,
    ("zoneinfo", "ZoneInfo"): lambda: __import__("zoneinfo").ZoneInfo,
}
_SUPPORTED_PICKLE_CLASS_GLOBALS: dict[PickleGlobal, set[PickleGlobal]] = {
    ("tensorflow.python.framework.ops", "EagerTensor"): {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("tensorflow.python.framework.ops", "convert_to_tensor"),
    },
    ("tensorflow.python.framework.ops", "_EagerTensorBase"): {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("tensorflow.python.framework.ops", "convert_to_tensor"),
    },
    ("tensorflow.python.framework.tensor", "Tensor"): {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("tensorflow.python.framework.ops", "convert_to_tensor"),
    },
    ("torch", "Tensor"): {
        ("collections", "OrderedDict"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch.storage", "_load_from_bytes"),
    },
    ("zoneinfo", "ZoneInfo"): {("zoneinfo", "ZoneInfo")},
    ("numpy", "ndarray"): {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
    },
    ("pandas.core.frame", "DataFrame"): {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
    },
}


def _safe_torch_storage_load_from_bytes(bs: bytes) -> t.Any:
    import torch

    f = io.BytesIO(bs)
    kwargs: dict[str, t.Any] = {}
    if not torch.cuda.is_available():
        kwargs["map_location"] = "cpu"
    try:
        return torch.load(f, weights_only=True, **kwargs)
    except TypeError:
        f.seek(0)
        return torch.load(f, **kwargs)


def build_safe_pickle_globals(
    allowed_classes: t.Iterable[type[t.Any]],
) -> dict[PickleGlobal, t.Any]:
    allowed_globals = dict(_SAFE_BUILTIN_GLOBALS)
    for cls in allowed_classes:
        key = (cls.__module__, cls.__name__)
        for global_key in _SUPPORTED_PICKLE_CLASS_GLOBALS.get(key, {key}):
            if global_key in allowed_globals:
                continue
            if global_key == key:
                allowed_globals[global_key] = cls
                continue
            factory = _OPTIONAL_SAFE_GLOBAL_FACTORIES.get(global_key)
            if factory is None:
                raise ValueError(
                    f"Unsupported safe pickle class: {cls.__module__}.{cls.__name__}"
                )
            allowed_globals[global_key] = factory()
    return allowed_globals


class SafeUnpickler(pickle.Unpickler):
    def __init__(
        self,
        file: t.BinaryIO,
        *,
        allowed_globals: dict[PickleGlobal, t.Any],
        **kwargs: t.Any,
    ) -> None:
        super().__init__(file, **kwargs)
        self._allowed_globals = allowed_globals

    def find_class(self, module: str, name: str) -> t.Any:
        key = (module, name)
        if key not in self._allowed_globals:
            raise pickle.UnpicklingError(f"global '{module}.{name}' is not allowed")
        return self._allowed_globals[key]


def safe_pickle_loads(
    data: bytes | memoryview,
    *,
    allowed_classes: t.Iterable[type[t.Any]] = (),
    buffers: t.Iterable[pickle.PickleBuffer] | None = None,
) -> t.Any:
    unpickler = SafeUnpickler(
        io.BytesIO(bytes(data)),
        allowed_globals=build_safe_pickle_globals(allowed_classes),
        buffers=buffers,
    )
    return unpickler.load()


def default_allowed_classes() -> tuple[type[t.Any], ...]:
    allowed_classes: list[type[t.Any]] = [
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        datetime.timezone,
        pathlib.Path,
        pathlib.PurePath,
        pathlib.PosixPath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.WindowsPath,
        Decimal,
        uuid.UUID,
    ]
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        allowed_classes.append(np.ndarray)
    try:
        import pandas as pd
    except ImportError:
        pass
    else:
        allowed_classes.append(pd.DataFrame)
    try:
        import zoneinfo
    except ImportError:
        pass
    else:
        allowed_classes.append(zoneinfo.ZoneInfo)
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        allowed_classes.append(Image.Image)
    try:
        import torch
    except ImportError:
        pass
    else:
        allowed_classes.append(torch.Tensor)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        allowed_classes.extend([tf.Tensor, type(tf.constant(0))])
    return tuple(allowed_classes)


PICKLE_SERDE_ALLOWED_CLASSES = default_allowed_classes()
