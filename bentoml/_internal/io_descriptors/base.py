import inspect
import typing as t
from abc import ABC, abstractmethod
from itertools import zip_longest
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from ..types import FileLike

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import PIL.Image  # noqa: F401


JSONType = t.Union[str, t.Dict[str, t.Any]]

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


def readable_str(obj: t.Any) -> t.Union[t.Dict[str, str], str]:
    # make str more human readable
    if callable(obj):
        return obj.__name__
    elif inspect.isclass(obj):
        return obj.__class__.__name__
    elif isinstance(obj, dict):
        fac = dict()  # type: t.Dict[str, t.Any]
        fac.update(zip(obj.keys(), map(readable_str, obj.values())))  # type: ignore
        return fac
    else:
        return str(obj)


class IODescriptor(ABC, t.Generic[IOPyObj]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
     in a bentoml.Service
    """

    HTTP_METHODS = ["POST"]
    __io_default_params__ = {}

    def __init__(self) -> None:
        if len(getattr(self, "__io_default_params__")) == 0:
            default_params: t.Dict[str, t.Any] = dict()
            init_fn = getattr(self, "__init__")
            spec = inspect.getfullargspec(init_fn)
            args = spec.args[1:]
            defaults = list(spec.defaults) if spec.defaults is not None else [None]
            if len(args) != 0:
                default_params.update(zip_longest(args, defaults))
            if spec.varkw is not None:
                default_params[spec.varkw] = spec.kwonlydefaults
            setattr(self, "__io_default_params__", default_params)

    def __str__(self) -> str:
        default_params = getattr(self, "__io_default_params__", None)
        filtered: t.List[str] = []
        for k, v in self.__dict__.items():
            if k.startswith("__"):
                continue
            key = k.strip("_")
            value = default_params.get(key, "")
            if isinstance(v, dict):
                val = readable_str(v)  # t.Dict[str, str]
                filtered = [f"{_k}={_v}" for _k, _v in val.items()]
                break
            if v != value:
                filtered.append(f"{key}={readable_str(v)}")
        return f"{self.__class__.__name__}({', '.join(filtered)})"

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
