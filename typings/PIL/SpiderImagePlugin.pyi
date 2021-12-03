"""
This type stub file was generated by pyright.
"""

from typing import Any
from typing_extensions import Literal
from .ImageFile import ImageFile

def isInt(f: object) -> Literal[0, 1]: ...

iforms: Any

def isSpiderHeader(t): ...
def isSpiderImage(filename): ...

class SpiderImageFile(ImageFile):
    format: str
    format_description: str
    @property
    def n_frames(self): ...
    @property
    def is_animated(self): ...
    def tell(self): ...
    stkoffset: Any
    fp: Any
    def seek(self, frame) -> None: ...
    def convert2byte(self, depth: int = ...): ...
    def tkPhotoImage(self): ...

def loadImageSeries(filelist: Any | None = ...): ...
def makeSpiderHeader(im): ...
