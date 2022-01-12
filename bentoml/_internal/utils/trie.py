import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    TrieDataType: t.TypeAlias = "t.Union[str, t.Dict[str, Trie]]"


class Trie:
    data: "TrieDataType"
    contained: bool

    def __init__(self, val: t.Optional[str] = None):
        if val is None:
            self.contained = False
            self.data = {}
        elif val == "":
            self.contained = True
            self.data = {}
        else:
            self.contained = False
            self.data = val

    def __repr__(self):  # pragma: no cover (implementation detail)
        return (self.contained, self.data).__repr__()

    def insert(self, val: str) -> int:
        if val == "":
            self.contained = True
            return 0

        if isinstance(self.data, str):
            self.data = {self.data[0]: Trie(self.data[1:])}

        if val[0] in self.data:
            return 1 + self.data[val[0]].insert(val[1:])
        else:
            self.data[val[0]] = Trie(val[1:])
            return 1

    def contains(self, val: str) -> bool:
        if val == "":
            return self.contained
        if isinstance(self.data, str):
            return self.data == val

        if not val[0] in self.data:
            return False

        return self.data[val[0]].contains(val[1:])
