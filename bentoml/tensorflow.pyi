import sys as _sys
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Union as _Union
from typing import overload as _overload

if _sys.version_info >= (3, 8):
    from typing import Literal as _Literal
else:
    from typing_extensions import Literal as _Literal

import numpy as _np
import tensorflow as _tf
import tensorflow.keras as _keras
import tensorflow_hub as _hub
from _internal.models import ModelStore as _ModelStore
from simple_di import Provider as _Provider
from simple_di import inject as _inject
from tensorflow_hub import native_module as _native_module
from tensorflow_hub import resolve as _resolve

from ._internal.runner import Runner as _Runner
from ._internal.types import PathType as _PathType
from ._internal.types import Tag as _Tag

@_overload
@_inject
def load(
    tag: _Union[str, _Tag],
    tfhub_tags: _List[str] = ...,
    tfhub_options: _Any = ...,
    load_as_wrapper: bool = ...,
    model_store: _Provider["ModelStore"] = ...,
) -> _Any: ...
@_overload
@_inject
def load(
    tag: _Union[str, _Tag],
    tfhub_tags: None = ...,
    tfhub_options: None = ...,
    load_as_wrapper: None = ...,
    model_store: _Provider["ModelStore"] = ...,
) -> _Any: ...
@_overload
@_inject
def import_from_tfhub(
    identifier: str,
    name: str = ...,
    metadata: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def import_from_tfhub(
    identifier: _hub.KerasLayer,
    name: str = ...,
    metadata: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def import_from_tfhub(
    identifier: _hub.Module,
    name: str = ...,
    metadata: None = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def import_from_tfhub(
    identifier: _Union[str, _hub.KerasLayer, _hub.Module],
    name: None = ...,
    metadata: None = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def save(
    name: str,
    model: _keras.Model,
    *,
    signatures: _Callable[..., _Any] = ...,
    options: _tf.saved_model.SaveOptions = ...,
    metadata: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def save(
    name: str,
    model: _tf.Module,
    *,
    signatures: _Dict[str, _Any] = ...,
    options: None = ...,
    metadata: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def save(
    name: str,
    model: _PathType,
    *,
    signatures: _Union[_Callable[..., _Any], _Dict[str, _Any]] = ...,
    options: _Optional[_tf.saved_model.SaveOptions] = ...,
    metadata: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def save(
    name: str,
    model: _Union["_keras.Model", "tf.Module", _PathType],
    *,
    signatures: _Union[_Callable[..., _Any], _Dict[str, _Any]] = ...,
    options: None = ...,
    metadata: None = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...

class _TensorflowRunner(_Runner):
    @_overload
    @_inject
    def __init__(
        self,
        tag: _Union[str, _Tag],
        predict_fn_name: str,
        device_id: str,
        partial_kwargs: _Dict[str, _Any],
        resource_quota: _Dict[str, _Any],
        batch_options: _Dict[str, _Any],
        model_store: _Provider["_ModelStore"] = ...,
    ) -> None: ...
    @_overload
    @_inject
    def __init__(
        self,
        tag: _Union[str, _Tag],
        predict_fn_name: str,
        device_id: str,
        partial_kwargs: None = ...,
        resource_quota: None = ...,
        batch_options: None = ...,
        model_store: _Provider["_ModelStore"] = ...,
    ) -> None: ...
    @property
    def required_models(self) -> _List[_Tag]: ...
    @property
    def num_concurrency_per_replica(self) -> int: ...
    @property
    def num_replica(self) -> int: ...
    def _setup(self) -> None: ...
    def _run_batch(
        self,
        *args: _Union[
            _List[_Union[int, float]], "_np.ndarray[_Any, _np.dtype[_Any]]", _tf.Tensor
        ],
        **kwargs: _Any,
    ) -> "_np.ndarray[_Any, _np.dtype[_Any]]": ...

@_overload
@_inject
def load_runner(
    tag: _Union[str, _Tag],
    *,
    predict_fn_name: str = ...,
    device_id: str = ...,
    partial_kwargs: _Dict[str, _Any] = ...,
    resource_quota: _Dict[str, _Any] = ...,
    batch_options: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> "_TensorflowRunner": ...
@_overload
@_inject
def load_runner(
    tag: _Union[str, _Tag],
    *,
    predict_fn_name: _Literal["__call__"] = ...,
    device_id: _Literal["CPU:0"] = ...,
    partial_kwargs: None = ...,
    resource_quota: None = ...,
    batch_options: None = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> "_TensorflowRunner": ...
