from __future__ import annotations

import io
import logging
import os
import sys
import typing as t
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from types import TracebackType
from typing import TYPE_CHECKING

if sys.version_info < (3, 8):
    import collections

    GenericClass = type(t.List)

    BUILTINS_MAPPING = {
        t.List: list,
        t.Set: set,
        t.Dict: dict,
        t.Tuple: tuple,
        t.ByteString: bytes,  # https://docs.python.org/3/library/typing.html#typing.ByteString
        t.Callable: collections.abc.Callable,
        t.Sequence: collections.abc.Sequence,
        type(None): None,
    }

    def _normalize_aliases(type_: t.Type) -> t.Type:
        if isinstance(type_, t.TypeVar):
            return type_

        if type_ in BUILTINS_MAPPING:
            return BUILTINS_MAPPING[type_]
        return type_

    def get_args(type_: t.Type) -> tuple[t.Type]:
        if isinstance(type_, GenericClass) and not type_._special:
            res = type_.__args__
            if get_origin(type_) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
        else:
            res = ()

        return res

    def get_origin(type_: t.Type) -> t.Type:
        if isinstance(type_, GenericClass) and not type_._special:
            ori = type_.__origin__
        elif hasattr(type_, "_special") and type_._special:
            ori = type_
        elif type_ is t.Generic:
            ori = t.Generic
        else:
            ori = None
        return ori

else:
    from typing import get_args
    from typing import get_origin

__all__ = [
    "MetadataType",
    "MetadataDict",
    "JSONSerializable",
    "LazyType",
    "is_compatible_type",
    "FileLike",
]

logger = logging.getLogger(__name__)

BATCH_HEADER = "Bentoml-Is-Batch-Request"

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = "latin1"

JSON_CHARSET = "utf-8"

if TYPE_CHECKING:
    PathType: t.TypeAlias = str | os.PathLike[str]
else:
    PathType = t.Union[str, os.PathLike]

MetadataType: t.TypeAlias = t.Union[
    str,
    bytes,
    bool,
    int,
    float,
    complex,
    datetime,
    date,
    time,
    timedelta,
    t.List["MetadataType"],
    t.Tuple["MetadataType"],
    t.Dict[str, "MetadataType"],
]

if TYPE_CHECKING:
    MetadataDict = t.Dict[str, MetadataType]
    JSONSerializable: t.TypeAlias = (
        str
        | int
        | float
        | bool
        | None
        | list["JSONSerializable"]
        | dict[str, "JSONSerializable"]
    )

    class ModelSignatureDict(t.TypedDict, total=False):
        batchable: bool
        batch_dim: tuple[int, int] | int | None
        input_spec: tuple[AnyType] | AnyType | None
        output_spec: AnyType | None

else:
    # NOTE: remove this when registering hook for MetadataType
    MetadataDict = dict
    ModelSignatureDict = dict

    JSONSerializable = t.NewType("JSONSerializable", object)

LifecycleHook = t.Callable[[], t.Union[None, t.Coroutine[t.Any, t.Any, None]]]

T = t.TypeVar("T")


class LazyType(t.Generic[T]):
    """
    LazyType provides solutions for several conflicts when applying lazy dependencies,
        type annotations and runtime class checking.
    It works both for runtime and type checking phases.

    * conflicts 1

    isinstance(obj, class) requires importing the class first, which breaks
    lazy dependencies

    solution:
    >>> LazyType("numpy.ndarray").isinstance(obj)

    * conflicts 2

    `isinstance(obj, str)` will narrow obj types down to str. But it only works for the
    case that the class is the type at the same time. For numpy.ndarray which the type
    is actually numpy.typing.NDArray, we had to hack the type checking.

    solution:
    >>> if TYPE_CHECKING:
    >>>     from numpy.typing import NDArray
    >>> LazyType["NDArray"]("numpy.ndarray").isinstance(obj)`
    >>> #  this will narrow the obj to NDArray with PEP-647

    * conflicts 3

    compare/refer/map classes before importing them.

    >>> HANDLER_MAP = {
    >>>     LazyType("numpy.ndarray"): ndarray_handler,
    >>>     LazyType("pandas.DataFrame"): pdframe_handler,
    >>> }
    >>>
    >>> HANDLER_MAP[LazyType(numpy.ndarray)]](array)
    >>> LazyType("numpy.ndarray") == numpy.ndarray
    """

    @t.overload
    def __init__(self, module_or_cls: str, qualname: str) -> None:
        """LazyType("numpy", "ndarray")"""

    @t.overload
    def __init__(self, module_or_cls: t.Type[T]) -> None:
        """LazyType(numpy.ndarray)"""

    @t.overload
    def __init__(self, module_or_cls: str) -> None:
        """LazyType("numpy.ndarray")"""

    def __init__(
        self,
        module_or_cls: str | t.Type[T],
        qualname: str | None = None,
    ) -> None:
        if isinstance(module_or_cls, str):
            if qualname is None:  # LazyType("numpy.ndarray")
                parts = module_or_cls.rsplit(".", 1)
                if len(parts) == 1:
                    raise ValueError("LazyType only works with classes")
                self.module, self.qualname = parts
            else:  # LazyType("numpy", "ndarray")
                self.module = module_or_cls
                self.qualname = qualname
            self._runtime_class = None
        else:  # LazyType(numpy.ndarray)
            self._runtime_class = module_or_cls
            self.module = module_or_cls.__module__
            if hasattr(module_or_cls, "__qualname__"):
                self.qualname: str = getattr(module_or_cls, "__qualname__")
            else:
                self.qualname: str = getattr(module_or_cls, "__name__")

    def __instancecheck__(self, obj: object) -> t.TypeGuard[T]:
        return self.isinstance(obj)

    @classmethod
    def from_type(cls, typ_: t.Union[LazyType[T], t.Type[T]]) -> LazyType[T]:
        if isinstance(typ_, LazyType):
            return typ_
        return cls(typ_)

    def __eq__(self, o: object) -> bool:
        """
        LazyType("numpy", "ndarray") == np.ndarray
        """
        if isinstance(o, type):
            o = self.__class__(o)

        if isinstance(o, LazyType):
            return self.module == o.module and self.qualname == o.qualname

        return False

    def __hash__(self) -> int:
        return hash(f"{self.module}.{self.qualname}")

    def __repr__(self) -> str:
        return f'LazyType("{self.module}", "{self.qualname}")'

    def get_class(self, import_module: bool = True) -> t.Type[T]:
        if self._runtime_class is None:
            try:
                m = sys.modules[self.module]
            except KeyError:
                if import_module:
                    import importlib

                    m = importlib.import_module(self.module)
                else:
                    raise ValueError(f"Module {self.module} not imported")

            self._runtime_class = t.cast("t.Type[T]", getattr(m, self.qualname))

        return self._runtime_class

    def isinstance(self, obj: t.Any) -> t.TypeGuard[T]:
        try:
            return isinstance(obj, self.get_class(import_module=False))
        except ValueError:
            return False


if TYPE_CHECKING:
    from types import UnionType

    AnyType: t.TypeAlias = t.Type[t.Any] | UnionType | LazyType[t.Any]
else:
    AnyType = t.Any


def is_compatible_type(t1: AnyType, t2: AnyType) -> bool:
    """
    A very loose check that it is possible for an object to be both an instance of ``t1``
    and an instance of ``t2``.

    Note: this will resolve ``LazyType``s, so should not be used in any
    peformance-critical contexts.
    """
    if get_origin(t1) is t.Union:
        return any((is_compatible_type(t2, arg_type) for arg_type in get_args(t1)))

    if get_origin(t2) is t.Union:
        return any((is_compatible_type(t1, arg_type) for arg_type in get_args(t2)))

    if isinstance(t1, LazyType):
        t1 = t1.get_class()

    if isinstance(t2, LazyType):
        t2 = t2.get_class()

    if isinstance(t1, type) and isinstance(t2, type):
        return issubclass(t1, t2) or issubclass(t2, t1)

    # catchall return true in unsupported cases so we don't error on unsupported types
    return True


@dataclass(frozen=False)
class FileLike(t.Generic[t.AnyStr], io.IOBase):
    """
    A wrapper for file-like objects that includes a custom name.
    """

    _wrapped: t.IO[t.AnyStr]
    _name: str

    @property
    def mode(self) -> str:
        return self._wrapped.mode

    @property
    def name(self) -> str:
        return self._name

    def close(self):
        self._wrapped.close()

    @property
    def closed(self) -> bool:
        return self._wrapped.closed

    def fileno(self) -> int:
        return self._wrapped.fileno()

    def flush(self):
        self._wrapped.flush()

    def isatty(self) -> bool:
        return self._wrapped.isatty()

    def read(self, size: int = -1) -> t.AnyStr:  # type: ignore # pylint: disable=arguments-renamed # python IO types
        return self._wrapped.read(size)

    def readable(self) -> bool:
        return self._wrapped.readable()

    def readline(self, size: int = -1) -> t.AnyStr:  # type: ignore (python IO types)
        return self._wrapped.readline(size)

    def readlines(self, size: int = -1) -> t.List[t.AnyStr]:  # type: ignore (python IO types)
        return self._wrapped.readlines(size)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._wrapped.seek(offset, whence)

    def seekable(self) -> bool:
        return self._wrapped.seekable()

    def tell(self) -> int:
        return self._wrapped.tell()

    def truncate(self, size: t.Optional[int] = None) -> int:
        return self._wrapped.truncate(size)

    def writable(self) -> bool:
        return self._wrapped.writable()

    def write(self, s: t.AnyStr) -> int:  # type: ignore (python IO types)
        return self._wrapped.write(s)

    def writelines(self, lines: t.Iterable[t.AnyStr]):  # type: ignore (python IO types)
        return self._wrapped.writelines(lines)

    def __next__(self) -> t.AnyStr:  # type: ignore (python IO types)
        return next(self._wrapped)

    def __iter__(self) -> t.Iterator[t.AnyStr]:  # type: ignore (python IO types)
        return self._wrapped.__iter__()

    def __enter__(self) -> t.IO[t.AnyStr]:
        return self._wrapped.__enter__()

    def __exit__(  # type: ignore (override python IO types)
        self,
        typ: t.Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._wrapped.__exit__(typ, value, traceback)
