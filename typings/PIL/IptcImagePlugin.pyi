"""
This type stub file was generated by pyright.
"""

from typing import Any

from .ImageFile import ImageFile

COMPRESSION: Any
PAD: Any

def i(c): ...
def dump(c) -> None: ...

class IptcImageFile(ImageFile):
    format: str
    format_description: str
    def getint(self, key): ...
    def field(self): ...
    im: Any
    def load(self): ...

def getiptcinfo(im): ...
