from typing import Any

from .error import YAMLError
from .nodes import Node

class SerializerError(YAMLError): ...

class Serializer:
    ANCHOR_TEMPLATE: Any
    use_encoding: Any
    use_explicit_start: Any
    use_explicit_end: Any
    use_version: Any
    use_tags: Any
    serialized_nodes: Any
    anchors: Any
    last_anchor_id: Any
    closed: Any

    def __init__(
        self,
        encoding: str = ...,
        explicit_start: bool | None = ...,
        explicit_end: bool | None = ...,
        version: bool | None = ...,
        tags: bool | None = ...,
    ) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def serialize(self, node: Node) -> None: ...
    def anchor_node(self, node: None) -> None: ...
    def generate_anchor(self, node: Node) -> Any: ...
    def serialize_node(self, node: Node, parent: Node, index: int) -> None: ...
