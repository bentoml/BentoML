import io
from pandas._typing import Buffer, CompressionOptions, FilePathOrBuffer, StorageOptions
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util._decorators import doc

class _XMLFrameParser:
    def __init__(
        self,
        path_or_buffer,
        xpath,
        namespaces,
        elems_only,
        attrs_only,
        names,
        encoding,
        stylesheet,
        compression,
        storage_options,
    ) -> None: ...
    def parse_data(self) -> list[dict[str, str | None]]: ...

class _EtreeFrameParser(_XMLFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...
    def parse_data(self) -> list[dict[str, str | None]]: ...

class _LxmlFrameParser(_XMLFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...
    def parse_data(self) -> list[dict[str, str | None]]: ...

def get_data_from_filepath(
    filepath_or_buffer, encoding, compression, storage_options
) -> str | bytes | Buffer: ...
def preprocess_data(data) -> io.StringIO | io.BytesIO: ...
@doc(storage_options=_shared_docs["storage_options"])
def read_xml(
    path_or_buffer: FilePathOrBuffer,
    xpath: str | None = ...,
    namespaces: dict | list[dict] | None = ...,
    elems_only: bool | None = ...,
    attrs_only: bool | None = ...,
    names: list[str] | None = ...,
    encoding: str | None = ...,
    parser: str | None = ...,
    stylesheet: FilePathOrBuffer | None = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...
