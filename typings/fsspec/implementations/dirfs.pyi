from typing import Any

from ..spec import AbstractFileSystem

class DirFileSystem(AbstractFileSystem):
    path: str
    fs: AbstractFileSystem
    def __init__(
        self,
        path: str | None = None,
        fs: AbstractFileSystem | None = None,
        fo: str | None = None,
        target_protocol: str | None = None,
        target_options: dict[str, Any] | None = None,
        **storage_options: Any,
    ) -> None: ...
