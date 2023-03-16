from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Literal
from typing import TypeVar
from typing import Protocol
from typing import Sequence
from pathlib import Path

from _typeshed import SupportsRead
from _typeshed import SupportsWrite
from numpy.typing import NDArray
from numpy.typing import DTypeLike

_Mode = Literal[
    "1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"
]
_Resample = Literal[0, 1, 2, 3, 4, 5]
_Size = Tuple[int, int]
_Box = Tuple[int, int, int, int]
_ConversionMatrix = (
    Union[
        Tuple[float, float, float, float],
        Tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ],
    ],
)
_Color = Union[float, Tuple[float, ...]]
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

class _Writeable(SupportsWrite[bytes], Protocol):
    def seek(self, __offset: int) -> Any: ...

NORMAL: Literal[0]
SEQUENCE: Literal[1]
CONTAINER: Literal[2]
MAX_IMAGE_PIXELS: int
NONE: Literal[0]
FLIP_LEFT_RIGHT: Literal[0]
FLIP_TOP_BOTTOM: Literal[1]
ROTATE_90: Literal[2]
ROTATE_180: Literal[3]
ROTATE_270: Literal[4]
TRANSPOSE: Literal[5]
TRANSVERSE: Literal[6]
AFFINE: Literal[0]
EXTENT: Literal[1]
PERSPECTIVE: Literal[2]
QUAD: Literal[3]
MESH: Literal[4]
NEAREST: Literal[0]
BOX: Literal[4]
BILINEAR: Literal[2]
LINEAR: Literal[2]
HAMMING: Literal[5]
BICUBIC: Literal[3]
CUBIC: Literal[3]
LANCZOS: Literal[1]
ANTIALIAS: Literal[1]
ORDERED: Literal[1]
RASTERIZE: Literal[2]
FLOYDSTEINBERG: Literal[3]
WEB: Literal[0]
ADAPTIVE: Literal[1]
MEDIANCUT: Literal[0]
MAXCOVERAGE: Literal[1]
FASTOCTREE: Literal[2]
LIBIMAGEQUANT: Literal[3]
ID: list[str]
OPEN: dict[str, Any]
MIME: dict[str, str]
SAVE: dict[str, Any]
SAVE_ALL: dict[str, Any]
EXTENSION: dict[str, str]
DECODERS: dict[str, Any]
ENCODERS: dict[str, Any]
MODES: list[_Mode]

_ImageState = Tuple[Dict[str, Any], str, Tuple[int, int], Any, bytes]

def init() -> int: ...

class Image:
    format: Any
    format_description: Any
    im: Any
    mode: str
    palette: Any
    info: dict[Any, Any]
    readonly: int
    pyaccess: Any

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def size(self) -> tuple[int, int]: ...
    def __enter__(self) -> Image: ...
    def __exit__(self, *args: Any) -> None: ...
    def close(self) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __array__(self, dtype: DTypeLike = ...) -> Any: ...
    def __getstate__(self) -> _ImageState: ...
    def __setstate__(self, state: _ImageState) -> None: ...
    def tobytes(self, encoder_name: str = ..., *args: Any) -> bytes: ...
    def frombytes(self, data: bytes, decoder_name: str = ..., *args: Any) -> None: ...
    def load(self) -> None: ...
    def verify(self) -> None: ...
    def putdata(
        self, data: Sequence[int], scale: float = ..., offset: float = ...
    ) -> None: ...
    def resize(
        self,
        size: tuple[int, int],
        resample: _Resample | None = ...,
        box: tuple[float, float, float, float] | None = ...,
        reducing_gap: float | None = ...,
    ) -> Image: ...
    def reduce(
        self, factor: int | tuple[int, int] | list[int], box: _Box | None = ...
    ) -> Image: ...
    def rotate(
        self,
        angle: float,
        resample: _Resample = ...,
        expand: bool = ...,
        center: tuple[float, float] | None = ...,
        translate: tuple[float, float] | None = ...,
        fillcolor: _Color | None = ...,
    ) -> Image: ...
    def save(
        self,
        fp: str | bytes | Path | _Writeable,
        format: str | None = ...,
        *,
        save_all: bool = ...,
        bitmap_format: Literal["bmp", "png"] = ...,
        **params: Any,
    ) -> None: ...
    def seek(self, frame: int) -> None: ...
    def show(self, title: str | None = ..., command: str | None = ...) -> None: ...
    def split(self) -> Tuple[Image, ...]: ...
    def getchannel(self, channel: int | str) -> Image: ...
    def tell(self) -> int: ...
    def thumbnail(
        self,
        size: tuple[int, int],
        resample: _Resample = ...,
        reducing_gap: float = ...,
    ) -> None: ...

def new(
    mode: _Mode, size: tuple[int, int], color: float | Tuple[float, ...] | str = ...
) -> Image: ...
def frombytes(
    mode: _Mode, size: tuple[int, int], data: bytes, decoder_name: str = ..., *args: Any
) -> Image: ...
def frombuffer(
    mode: _Mode, size: tuple[int, int], data: bytes, decoder_name: str = ..., *args: Any
) -> Image: ...
def fromarray(obj: NDArray[Any], mode: _Mode | None = ...) -> Image: ...
def open(
    fp: str | bytes | Path | SupportsRead[bytes],
    mode: Literal["r"] = ...,
    formats: list[str] | tuple[str] | None = ...,
) -> Image: ...
