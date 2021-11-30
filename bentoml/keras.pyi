import sys as _sys
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Union as _Union
from typing import overload as _overload

if _sys.version_info >= (3, 8):
    from typing import Literal as _Literal
else:
    from typing_extensions import Literal as _Literal

import numpy as _np
import tensorflow as _tf
import tensorflow.keras as _keras
from simple_di import Provider as _Provider
from simple_di import inject as _inject
from tensorflow.python.client.session import BaseSession as _BaseSession

from bentoml._internal.frameworks.tensorflow import (
    _TensorflowRunner,  # type: ignore[reportPrivateUsage]
)

from ._internal.models import ModelStore as _ModelStore
from ._internal.types import Tag as _Tag

def get_session() -> "_BaseSession": ...
@_inject
def load(
    tag: _Union[str, _Tag],
    model_store: _Provider["_ModelStore"] = ...,
) -> "_keras.Model": ...
@_overload
@_inject
def save(
    name: str,
    model: "_keras.Model",
    *,
    store_as_json: bool = ...,
    custom_objects: _Dict[str, _Any] = ...,
    metadata: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...
@_overload
@_inject
def save(
    name: str,
    model: "_keras.Model",
    *,
    store_as_json: _Literal[False] = ...,
    custom_objects: None = ...,
    metadata: None = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> _Tag: ...

class _KerasRunner(_TensorflowRunner):
    @_overload
    @_inject
    def __init__(
        self,
        tag: _Union[str, _Tag],
        predict_fn_name: str,
        device_id: str,
        predict_kwargs: _Dict[str, _Any],
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
        predict_kwargs: None = ...,
        resource_quota: None = ...,
        batch_options: None = ...,
        model_store: _Provider["_ModelStore"] = ...,
    ) -> None: ...
    def _setup(self) -> None: ...
    def _run_batch(self, input_data: _Union[List[_Union[int, float]], "_np.ndarray[_Any, _np.dtype[_Any]]", _tf.Tensor]) -> _Union["_np.ndarray[_Any, _np.dtype[_Any]]", _tf.Tensor]: ...  # type: ignore[override]

@_overload
@_inject
def load_runner(
    tag: _Union[str, _Tag],
    *,
    predict_fn_name: str = ...,
    device_id: str = ...,
    predict_kwargs: _Dict[str, _Any] = ...,
    resource_quota: _Dict[str, _Any] = ...,
    batch_options: _Dict[str, _Any] = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> "_KerasRunner": ...
@_overload
@_inject
def load_runner(
    tag: _Union[str, _Tag],
    *,
    predict_fn_name: _Literal["predict"] = ...,
    device_id: _Literal["CPU:0"] = ...,
    predict_kwargs: None = ...,
    resource_quota: None = ...,
    batch_options: None = ...,
    model_store: _Provider["_ModelStore"] = ...,
) -> "_KerasRunner": ...
