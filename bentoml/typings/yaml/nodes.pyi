from typing import Any
from typing import ClassVar

from yaml.error import Mark

class Node:
    tag: str
    value: Any
    start_mark: Mark | Any
    end_mark: Mark | Any

    def __init__(
        self, tag: str, value: Any, start_mark: Mark | None, end_mark: Mark | None
    ) -> None: ...

class ScalarNode(Node):
    id: ClassVar[str]
    style: str | Any

    def __init__(
        self,
        tag: str,
        value: Any,
        start_mark: Mark | None = ...,
        end_mark: Mark | None = ...,
        style: str | None = ...,
    ) -> None: ...

class CollectionNode(Node):
    flow_style: bool | Any

    def __init__(
        self,
        tag: str,
        value: Any,
        start_mark: Mark | None = ...,
        end_mark: Mark | None = ...,
        flow_style: bool | None = ...,
    ) -> None: ...

class SequenceNode(CollectionNode): ...
class MappingNode(CollectionNode): ...
