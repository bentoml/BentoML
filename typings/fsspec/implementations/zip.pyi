from typing import Any
from typing import BinaryIO
from zipfile import ZipFile

from ..spec import AbstractFileSystem

class ZipFileSystem(AbstractFileSystem):
    mode: str
    of: BinaryIO
    fo: BinaryIO
    force_zip_64: bool
    zip: ZipFile
    def __init__(
        self,
        fo: str = "",
        mode: str = "r",
        target_protocol: str | None = None,
        target_options: dict[str, Any] | None = None,
        compression: int = ...,
        allowZip64: bool = True,
        compresslevel: int | None = None,
        **kwargs: Any,
    ) -> None: ...
