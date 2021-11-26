from io import BytesIO, BufferedWriter
from pickle import PickleBuffer
from typing import Any, Optional, overload
from cloudpickle.cloudpickle import *
from cloudpickle.cloudpickle_fast import CloudPickler

@overload
def dump(
    obj: Any,
    file: BytesIO,
    protocol: Optional[int] = ...,
    buffer_callback: Callable[[PickleBuffer], Any] = ...,
) -> None: ...
@overload
def dump(
    obj: Any,
    file: BufferedWriter,
    protocol: Optional[int] = ...,
    buffer_callback: Callable[[PickleBuffer], Any] = ...,
) -> None: ...
@overload
def dump(obj: Any, file: BytesIO, protocol: Optional[int] = ...) -> None: ...
@overload
def dump(obj: Any, file: BufferedWriter, protocol: Optional[int] = ...) -> None: ...
@overload
def dumps(
    obj: Any,
    file: BytesIO,
    protocol: Optional[int] = ...,
    buffer_callback: Callable[[PickleBuffer], Any] = ...,
) -> None: ...
@overload
def dumps(
    obj: Any,
    file: BufferedWriter,
    protocol: Optional[int] = ...,
    buffer_callback: Callable[[PickleBuffer], Any] = ...,
) -> None: ...
@overload
def dumps(obj: Any, protocol: Optional[int] = ...) -> None: ...
@overload
def dumps(obj: Any, protocol: Optional[int] = ...) -> None: ...

Pickler = CloudPickler
__version__: str = ...
