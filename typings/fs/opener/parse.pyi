import collections, typing
from typing import Text

if typing.TYPE_CHECKING: ...

class ParseResult(
    collections.namedtuple(
        "ParseResult",
        ["protocol", "username", "password", "resource", "params", "path"],
    )
): ...

_RE_FS_URL = ...

def parse_fs_url(fs_url: Text) -> ParseResult: ...
