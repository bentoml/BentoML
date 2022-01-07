import json
import typing as t
import dataclasses
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import JSONType
from .base import IODescriptor
from ..types import LazyType
from ...exceptions import BadInput
from ...exceptions import MissingDependencyException

if TYPE_CHECKING:
    import pydantic

    from .. import ext_typing as ext

    _SerializableObj = t.Union[
        "ext.NpNDArray[t.Any]",
        "ext.PdDataFrame",
        t.Type["pydantic.BaseModel"],
        t.Any,
    ]


MIME_TYPE_JSON = "application/json"


class DefaultJsonEncoder(json.JSONEncoder):  # pragma: no cover
    def default(self, o: _SerializableObj) -> t.Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if LazyType["ext.NpNDArray[t.Any]"]("numpy.ndarray").isinstance(o):
            return o.tolist()
        if LazyType["ext.NpGeneric"]("numpy.generic").isinstance(o):
            return o.item()
        if LazyType["ext.PdDataFrame"]("pandas.DataFrame").isinstance(o):
            return o.to_dict()  # type: ignore[attr-defined]
        if LazyType["ext.PdSeries"]("pandas.Series").isinstance(o):
            return o.to_dict()  # type: ignore[attr-defined]
        if LazyType["pydantic.BaseModel"]("pydantic.BaseModel").isinstance(o):
            obj_dict = o.dict()
            if "__root__" in obj_dict:
                obj_dict = obj_dict.get("__root__")
            return obj_dict

        return super().default(o)


class JSON(IODescriptor[JSONType]):
    """
    `JSON` defines API specification for the inputs/outputs of a Service, where either
     inputs will be converted to or outputs will be converted from a JSON representation
     as specified in your API function signature.

    .. Toy implementation of a sklearn service::
        # sklearn_svc.py
        import pandas as pd
        import numpy as np
        from bentoml.io import PandasDataFrame, JSON
        import bentoml.sklearn

        input_spec = PandasDataFrame.from_sample(pd.DataFrame(np.array([[5,4,3,2]])))

        runner = bentoml.sklearn.load_runner("sklearn_model_clf")

        svc = bentoml.Service("iris-classifier", runners=[runner])

        @svc.api(input=input_spec, output=JSON())
        def predict(input_arr: pd.DataFrame):
            res = runner.run_batch(input_arr)  # type: np.ndarray
            return {"res":pd.DataFrame(res).to_json(orient='record')}

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "iris-classifier" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        % curl -X POST -H "Content-Type: application/json" --data
          '[{"0":5,"1":4,"2":3,"3":2}]' http://0.0.0.0:5000/predict

        {"res":"[{\"0\":1}]"}%

    Args:
        pydantic_model (`pydantic.BaseModel`, `optional`, default to `None`):
            Pydantic model schema.
        validate_json (`bool`, `optional`, default to `True`):
            If True, then use Pydantic model specified above to validate given JSON.
        json_encoder (`Type[json.JSONEncoder]`, default to
                      `~bentoml._internal.io_descriptor.json.DefaultJsonEncoder`):
            JSON encoder.
    Returns:
        IO Descriptor that in JSON format.
    """

    def __init__(
        self,
        pydantic_model: t.Optional["pydantic.BaseModel"] = None,
        validate_json: bool = True,
        json_encoder: t.Type[json.JSONEncoder] = DefaultJsonEncoder,
    ):
        if pydantic_model is not None:  # pragma: no cover
            try:
                import pydantic as _
            except ImportError:
                raise MissingDependencyException(
                    "`pydantic` must be installed to use `pydantic_model`"
                )
            self._pydantic_model = pydantic_model

        self._validate_json = validate_json
        self._json_encoder = json_encoder

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        if hasattr(self, "_pydantic_model"):
            return self._pydantic_model.schema()
        return {"type": "object"}

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    async def from_http_request(self, request: Request) -> JSONType:
        json_obj = await request.json()
        if hasattr(self, "_pydantic_model") and self._validate_json:
            try:
                import pydantic
            except ImportError:
                raise MissingDependencyException(
                    "`pydantic` must be installed to use `pydantic_model`"
                )
            try:
                return self._pydantic_model.parse_obj(json_obj)
            except pydantic.ValidationError:
                raise BadInput("Invalid JSON Request received")
        return json_obj

    async def to_http_response(self, obj: JSONType) -> Response:
        json_str = json.dumps(
            obj,
            cls=self._json_encoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        )
        return Response(json_str, media_type=MIME_TYPE_JSON)
