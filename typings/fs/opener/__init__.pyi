import contextlib, typing as t
from ..base import FS
from .base import Opener
from .parse import parse_fs_url as parse
from .registry import registry

if t.TYPE_CHECKING: ...

def open_fs(
    fs_url: t.Union[FS, t.Text],
    writeable: bool = ...,
    create: bool = ...,
    cwd: t.Text = ...,
    default_protocol: t.Text = ...,
) -> FS: ...
def open(
    fs_url: t.Text,
    writeable: bool = ...,
    create: bool = ...,
    cwd: t.Text = ...,
    default_protocol: t.Text = ...,
) -> t.Tuple[FS, t.Text]: ...
@contextlib.contextmanager
def manage_fs(
    fs_url: t.Union[FS, t.Text],
    create: bool = ...,
    writeable: bool = ...,
    cwd: t.Text = ...,
) -> t.Iterator[FS]: ...

__all__ = ["registry", "Opener", "open_fs", "open", "manage_fs", "parse"]
