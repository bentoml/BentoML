import typing
from typing import List, Optional, Text, TextIO, Tuple
from .base import FS

if typing.TYPE_CHECKING: ...

def render(
    fs: FS,
    path: Text = ...,
    file: Optional[TextIO] = ...,
    encoding: Optional[Text] = ...,
    max_levels: int = ...,
    with_color: Optional[bool] = ...,
    dirs_first: bool = ...,
    exclude: Optional[List[Text]] = ...,
    filter: Optional[List[Text]] = ...,
) -> Tuple[int, int]: ...
