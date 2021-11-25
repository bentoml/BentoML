

import typing
from typing import Text

if typing.TYPE_CHECKING: ...
_WINDOWS_PLATFORM = ...

def url_quote(path_snippet: Text) -> Text:
    """Quote a URL without quoting the Windows drive letter, if any.

    On Windows, it will separate drive letter and quote Windows
    path alone. No magic on Unix-like path, just pythonic
    `~urllib.request.pathname2url`.

    Arguments:
       path_snippet (str): a file path, relative or absolute.

    """
    ...
