from ._functools import method_cache

class FoldedCase(str):
    def __lt__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __contains__(self, other): ...
    def in_(self, other): ...
    @method_cache
    def lower(self): ...
    def index(self, sub): ...
    def split(self, splitter=..., maxsplit=...): ...
