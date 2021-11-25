

import typing
from typing import List, Optional, Text, TextIO, Tuple

from .base import FS

"""Render a FS object as text tree views.

Color is supported on UNIX terminals.
"""
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
) -> Tuple[int, int]:
    """Render a directory structure in to a pretty tree.

    Arguments:
        fs (~fs.base.FS): A filesystem instance.
        path (str): The path of the directory to start rendering
            from (defaults to root folder, i.e. ``'/'``).
        file (io.IOBase): An open file-like object to render the
            tree, or `None` for stdout.
        encoding (str, optional): Unicode encoding, or `None` to
            auto-detect.
        max_levels (int, optional): Maximum number of levels to
            display, or `None` for no maximum.
        with_color (bool, optional): Enable terminal color output,
            or `None` to auto-detect terminal.
        dirs_first (bool): Show directories first.
        exclude (list, optional): Option list of directory patterns
            to exclude from the tree render.
        filter (list, optional): Optional list of files patterns to
            match in the tree render.

    Returns:
        (int, int): A tuple of ``(<directory count>, <file count>)``.

    """
    ...
