import inspect
import typing as t
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from ..types import FileLike

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import PIL.Image
    from pydantic import BaseModel


JSONType = t.Union[str, t.Dict[str, t.Any], "BaseModel"]

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


def _mk_str(obj: t.Any) -> str:
    # make str more human readable
    if callable(obj):
        return obj.__name__
    elif inspect.isclass(obj):
        return obj.__class__.__name__
    elif isinstance(obj, dict):
        fac = dict()  # type: t.Dict[str, t.Any]
        fac.update(zip(obj.keys(), map(_mk_str, obj.values())))  # type: ignore
        return str(fac)
    else:
        return str(obj)


class IODescriptor(ABC, t.Generic[IOPyObj]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
     in a bentoml.Service
    """

    HTTP_METHODS = ["POST"]

    def __str__(self) -> str:
        return f"%s(%s)" % (
            self.__class__.__name__,
            ", ".join(
                [f'{k.strip("_")}={_mk_str(v)}' for k, v in self.__dict__.items()]
            ),
        )

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
