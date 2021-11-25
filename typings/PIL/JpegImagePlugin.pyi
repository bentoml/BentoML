

from typing import Any

from .ImageFile import ImageFile

def Skip(self, marker) -> None: ...
def APP(self, marker) -> None: ...
def COM(self, marker) -> None: ...
def SOF(self, marker) -> None: ...
def DQT(self, marker) -> None: ...

MARKER: Any

class JpegImageFile(ImageFile):
    format: str
    format_description: str
    def load_read(self, read_bytes): ...
    mode: Any
    tile: Any
    decoderconfig: Any
    def draft(self, mode, size): ...
    im: Any
    def load_djpeg(self) -> None: ...
    def getxmp(self): ...

RAWMODE: Any
zigzag_index: Any
samplings: Any

def convert_dict_qtables(qtables): ...
def get_sampling(im): ...
def jpeg_factory(fp: Any | None = ..., filename: Any | None = ...): ...
