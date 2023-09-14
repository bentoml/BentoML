from __future__ import annotations

from typing import Text
from typing import NamedTuple

class ParseResult(NamedTuple):
    protocol: Text
    username: Text
    password: Text
    resource: Text
    params: dict[str, str]
    path: Text

def parse_fs_url(fs_url: Text) -> ParseResult: ...
