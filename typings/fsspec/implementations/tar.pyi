from tarfile import TarFile
from typing import Any
from typing import BinaryIO

from ..spec import AbstractFileSystem

class TarFileSystem(AbstractFileSystem):
    fo: BinaryIO
    tar: TarFile
    def __init__(
        self,
        fo: str = "",
        index_store: None = None,
        target_options: dict[str, Any] | None = None,
        target_protocol: str | None = None,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None: ...
