import typing
from typing import List, Text, Tuple

if typing.TYPE_CHECKING: ...
__all__ = [
    "abspath",
    "basename",
    "combine",
    "dirname",
    "forcedir",
    "frombase",
    "isabs",
    "isbase",
    "isdotfile",
    "isparent",
    "issamedir",
    "iswildcard",
    "iteratepath",
    "join",
    "normpath",
    "parts",
    "recursepath",
    "relativefrom",
    "relpath",
    "split",
    "splitext",
]
_requires_normalization = ...

def normpath(path: Text) -> Text: ...
def iteratepath(path: Text) -> List[Text]: ...
def recursepath(path: Text, reverse: bool = ...) -> List[Text]: ...
def isabs(path: Text) -> bool: ...
def abspath(path: Text) -> Text: ...
def relpath(path: Text) -> Text: ...
def join(*paths: Text) -> Text: ...
def combine(path1: Text, path2: Text) -> Text: ...
def parts(path: Text) -> List[Text]: ...
def split(path: Text) -> Tuple[(Text, Text)]: ...
def splitext(path: Text) -> Tuple[(Text, Text)]: ...
def isdotfile(path: Text) -> bool: ...
def dirname(path: Text) -> Text: ...
def basename(path: Text) -> Text: ...
def issamedir(path1: Text, path2: Text) -> bool: ...
def isbase(path1: Text, path2: Text) -> bool: ...
def isparent(path1: Text, path2: Text) -> bool: ...
def forcedir(path: Text) -> Text: ...
def frombase(path1: Text, path2: Text) -> Text: ...
def relativefrom(base: Text, path: Text) -> Text: ...

_WILD_CHARS = ...

def iswildcard(path: Text) -> bool: ...
