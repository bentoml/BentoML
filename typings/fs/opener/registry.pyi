import contextlib
import typing
from typing import Callable
from typing import Iterator
from typing import List
from typing import Text
from typing import Tuple
from typing import Type
from typing import Union

from ..base import FS
from .base import Opener

if typing.TYPE_CHECKING: ...

class Registry:
    def __init__(self, default_opener: Text = ..., load_extern: bool = ...) -> None: ...
    def __repr__(self) -> Text: ...
    def install(
        self, opener: Union[Type[Opener], Opener, Callable[[], Opener]]
    ) -> Opener: ...
    @property
    def protocols(self) -> List[Text]: ...
    def get_opener(self, protocol: Text) -> Opener: ...
    def open(
        self,
        fs_url: Text,
        writeable: bool = ...,
        create: bool = ...,
        cwd: Text = ...,
        default_protocol: Text = ...,
    ) -> Tuple[FS, Text]: ...
    def open_fs(
        self,
        fs_url: Union[FS, Text],
        writeable: bool = ...,
        create: bool = ...,
        cwd: Text = ...,
        default_protocol: Text = ...,
    ) -> FS: ...
    @contextlib.contextmanager
    def manage_fs(
        self,
        fs_url: Union[FS, Text],
        create: bool = ...,
        writeable: bool = ...,
        cwd: Text = ...,
    ) -> Iterator[FS]: ...

registry: Registry = ...
