import io
import os
import sys
import typing as t
import urllib
import logging
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING
from dataclasses import dataclass

from .utils.dataclasses import json_serializer

if sys.version_info < (3, 7):
    from backports.datetime_fromisoformat import MonkeyPatch

    MonkeyPatch.patch_fromisoformat()

logger = logging.getLogger(__name__)

BATCH_HEADER = "Bentoml-Is-Batch-Request"

# For non latin1 characters: https://tools.ietf.org/html/rfc8187
# Also https://github.com/benoitc/gunicorn/issues/1778
HEADER_CHARSET = "latin1"

JSON_CHARSET = "utf-8"

if TYPE_CHECKING:
    PathType = t.Union[str, os.PathLike[str]]
else:
    PathType = t.Union[str, os.PathLike]

JSONSerializable = t.NewType("JSONSerializable", object)


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
        module_or_cls: t.Union[str, t.Type[T]],
        qualname: t.Optional[str] = None,
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

    @classmethod
    def from_type(cls, typ_: t.Union["LazyType[T]", "t.Type[T]"]) -> "LazyType[T]":
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

    def get_class(self, import_module: bool = True) -> "t.Type[T]":
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

    def isinstance(self, obj: t.Any) -> "t.TypeGuard[T]":
        try:
            return isinstance(obj, self.get_class(import_module=False))
        except ValueError:
            return False


@json_serializer(fields=["uri", "name"], compat=True)
@dataclass(frozen=False)
class FileLike:
    """
    An universal lazy-loading wrapper for file-like objects.
    It accepts URI, file path or bytes and provides interface like opened file object.

    Class attributes:

    - bytes (`bytes`, `optional`):
    - uri (:code:`str`, `optional`):
        The set of possible uris is:

        - :code:`file:///home/user/input.json`
        - :code:`http://site.com/input.csv` (Not implemented)
        - :code:`https://site.com/input.csv` (Not implemented)

    - name (:code:`str`, `optional`, default to :obj:`None`)

    """

    bytes_: t.Optional[bytes] = None
    uri: t.Optional[str] = None
    name: t.Optional[str] = None

    _stream: t.Optional[t.BinaryIO] = None

    def __post_init__(self):
        if self.name is None:
            if self._stream is not None:
                self.name = getattr(self._stream, "name", None)
            elif self.uri is not None:
                p = urllib.parse.urlparse(self.uri)  # type: ignore
                if p.scheme and p.scheme != "file":
                    raise NotImplementedError(
                        f"{self.__class__} now supports scheme 'file://' only"
                    )
                _, self.name = os.path.split(self.path)

    @property
    def path(self):
        r"""
        supports:

            /home/user/file
            C:\Python27\Scripts\pip.exe
            \\localhost\c$\WINDOWS\clock.avi
            \\networkstorage\homes\user

        .. note::
            https://stackoverflow.com/a/61922504/3089381
        """
        parsed = urllib.parse.urlparse(self.uri)  # type: ignore
        raw_path = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))  # type: ignore # noqa: LN001
        host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
        path = os.path.abspath(os.path.join(host, raw_path))
        return path

    @property
    def stream(self) -> t.BinaryIO:
        if self._stream is not None:
            pass
        elif self.bytes_ is not None:
            self._stream = io.BytesIO(self.bytes_)
        elif self.uri is not None:
            self._stream = open(self.path, "rb")
        else:
            return io.BytesIO()
        return self._stream

    def read(self, size: int = -1):
        # TODO: also write to log
        return self.stream.read(size)

    def seek(self, pos: int):
        return self.stream.seek(pos)

    def tell(self):
        return self.stream.tell()

    def close(self):
        if self._stream is not None:
            self._stream.close()
