import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from ..types import FileLike

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import pydantic  # noqa: F401
    import PIL.Image  # noqa: F401


JSONType = t.Union[str, t.Dict[str, t.Any], "pydantic.BaseModel"]


# NOTES: we will keep type in quotation to avoid backward compatibility
#  with numpy < 1.20, since we will use the latest stubs from the main branch of numpy.
#  that enable a new way to type hint an ndarray.
ImageType = t.Union["PIL.Image.Image", "np.ndarray[t.Any, np.dtype[t.Any]]"]

IOType = t.Union[
    str,
    JSONType,
    FileLike,
    ImageType,
    "np.ndarray[t.Any, np.dtype[t.Any]]",
    "pd.DataFrame",
    "pd.Series[t.Any]",
]


IOPyObj = t.TypeVar("IOPyObj")


_T = t.TypeVar("_T")


class IODescriptor(ABC, t.Generic[IOPyObj]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]
    _init_str: str = ""

    def __new__(cls: t.Type[_T], *args: t.Any, **kwargs: t.Any) -> _T:
        self = super().__new__(cls)
        arg_strs = tuple(repr(i) for i in args) + tuple(
            f"{k}={repr(v)}" for k, v in kwargs.items()
        )
        setattr(self, "_init_str", f"{cls.__name__}({', '.join(arg_strs)})")

        return self

    def __repr__(self) -> str:
        return self._init_str

    @abstractmethod
    def openapi_schema_type(self) -> t.Dict[str, str]:
        ...

    @abstractmethod
    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        ...

    @abstractmethod
    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        ...

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOPyObj:
        ...

    @abstractmethod
    async def to_http_response(self, obj: IOPyObj) -> Response:
        ...

    # TODO: gRPC support
    # @abstractmethod
    # def generate_protobuf(self): ...

    # @abstractmethod
    # async def from_grpc_request(self, request: GRPCRequest) -> IOType: ...

    # @abstractmethod
    # async def to_grpc_response(self, obj: IOType) -> GRPCResponse: ...
