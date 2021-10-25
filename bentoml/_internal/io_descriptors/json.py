import dataclasses
import json
import typing as t

from starlette.requests import Request
from starlette.responses import Response

from bentoml.exceptions import BadInput, InvalidArgument, MissingDependencyException

from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:
    import numpy as np
    import pydantic

    _major, _minor = list(map(lambda x: int(x), np.__version__.split(".")[:2]))
    if (_major, _minor) > (1, 20):
        from numpy.typing import ArrayLike
    else:
        from ..typing_extensions.numpy import ArrayLike

else:
    np = LazyLoader("np", globals(), "numpy")
    ArrayLike = t.Any

    try:
        import pydantic
    except ModuleNotFoundError:
        pydantic = None

MIME_TYPE_JSON = "application/json"

_SerializedObj = t.TypeVar(
    "_SerializedObj",
    bound=t.Union[
        "np.generic",
        ArrayLike,
        "pydantic.BaseModel",
        t.Any,
    ],
)


class DefaultJsonEncoder(json.JSONEncoder):
    def default(self, o: _SerializedObj) -> t.Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if isinstance(o, np.generic):
            return o.item()

        if isinstance(o, np.ndarray):
            return o.tolist()

        if pydantic and isinstance(o, pydantic.BaseModel):
            obj_dict = o.dict()
            if "__root__" in obj_dict:
                obj_dict = obj_dict["__root__"]
            return obj_dict

        return super().default(o)


class JSON(IODescriptor):
    def __init__(
        self,
        pydantic_model: t.Optional["pydantic.BaseModel"] = None,
        validate_json: bool = True,
        json_encoder: t.Optional[t.Type[json.JSONEncoder]] = DefaultJsonEncoder,
    ):
        if pydantic_model is not None:
            if pydantic is None:
                raise MissingDependencyException(
                    "`pydantic` must be installed to use `pydantic_model`"
                )

            if not isinstance(pydantic_model, pydantic.BaseModel):
                raise InvalidArgument(
                    "Invalid argument type 'pydantic_model' for JSON io descriptor,"
                    "must be an instance of a pydantic model"
                )
            self._pydantic_model = pydantic_model

        self._validate_json = validate_json
        self._json_encoder = json_encoder

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        pass

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        pass

    def openapi_schema(self) -> t.Dict[str, t.Dict[str, t.Dict[str, t.Any]]]:
        if self._pydantic_model:
            schema = self._pydantic_model.schema()
        else:
            schema = {"type": "object"}
        return {MIME_TYPE_JSON: {"schema": schema}}

    async def from_http_request(self, request: Request) -> t.Any:
        json_obj = await request.json()
        if self._pydantic_model and self._validate_json:
            try:
                return self._pydantic_model.parse_obj(json_obj)
            except pydantic.ValidationError:
                raise BadInput("Invalid JSON Request received")
        else:
            return json_obj

    async def to_http_response(self, obj: t.Any) -> Response:
        json_str = json.dumps(
            obj,
            cls=self._json_encoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        )
        return Response(json_str, media_type=MIME_TYPE_JSON)
