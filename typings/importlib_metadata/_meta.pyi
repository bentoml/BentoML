from typing import Any, Dict, Iterator, List, Union
from ._compat import Protocol

_T = ...

class PackageMetadata(Protocol):
    def __len__(self) -> int: ...
    def __contains__(self, item: str) -> bool: ...
    def __getitem__(self, key: str) -> str: ...
    def __iter__(self) -> Iterator[str]: ...
    def get_all(self, name: str, failobj: _T = ...) -> Union[(List[Any], _T)]: ...
    @property
    def json(self) -> Dict[(str, Union[(str, List[str])])]: ...

class SimplePath(Protocol):
    def joinpath(self) -> SimplePath: ...
    def __div__(self) -> SimplePath: ...
    def parent(self) -> SimplePath: ...
    def read_text(self) -> str: ...
