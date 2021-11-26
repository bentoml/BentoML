from typing import Any
from pandas._typing import CompressionOptions, FilePathOrBuffer, StorageOptions
from pandas.core.frame import DataFrame

class BaseXMLFormatter:
    def __init__(
        self,
        frame: DataFrame,
        path_or_buffer: FilePathOrBuffer | None = ...,
        index: bool | None = ...,
        root_name: str | None = ...,
        row_name: str | None = ...,
        na_rep: str | None = ...,
        attr_cols: list[str] | None = ...,
        elem_cols: list[str] | None = ...,
        namespaces: dict[str | None, str] | None = ...,
        prefix: str | None = ...,
        encoding: str = ...,
        xml_declaration: bool | None = ...,
        pretty_print: bool | None = ...,
        stylesheet: FilePathOrBuffer | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def build_tree(self) -> bytes: ...
    def validate_columns(self) -> None: ...
    def validate_encoding(self) -> None: ...
    def process_dataframe(self) -> dict[int | str, dict[str, Any]]: ...
    def handle_indexes(self) -> None: ...
    def get_prefix_uri(self) -> str: ...
    def other_namespaces(self) -> dict: ...
    def build_attribs(self) -> None: ...
    def build_elems(self) -> None: ...
    def write_output(self) -> str | None: ...

class EtreeXMLFormatter(BaseXMLFormatter):
    def __init__(self, *args, **kwargs) -> None: ...
    def build_tree(self) -> bytes: ...
    def get_prefix_uri(self) -> str: ...
    def build_attribs(self) -> None: ...
    def build_elems(self) -> None: ...
    def prettify_tree(self) -> bytes: ...
    def add_declaration(self) -> bytes: ...
    def remove_declaration(self) -> bytes: ...

class LxmlXMLFormatter(BaseXMLFormatter):
    def __init__(self, *args, **kwargs) -> None: ...
    def build_tree(self) -> bytes: ...
    def convert_empty_str_key(self) -> None: ...
    def get_prefix_uri(self) -> str: ...
    def build_attribs(self) -> None: ...
    def build_elems(self) -> None: ...
    def transform_doc(self) -> bytes: ...
