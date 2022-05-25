from __future__ import annotations

import json
import typing as t
import dataclasses
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from ..types import LazyType
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import MissingDependencyException

if TYPE_CHECKING:
    from types import UnionType

    import pydantic

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context

    _SerializableObj: t.TypeAlias = t.Union[
        "ext.NpNDArray",
        "ext.PdDataFrame",
        t.Type["pydantic.BaseModel"],
        t.Any,
    ]


JSONType = t.Union[str, t.Dict[str, t.Any], "pydantic.BaseModel"]

MIME_TYPE_JSON = "application/json"


class DefaultJsonEncoder(json.JSONEncoder):  # pragma: no cover
    def default(self, o: _SerializableObj) -> t.Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(o):
            return o.tolist()
        if LazyType["ext.NpGeneric"]("numpy.generic").isinstance(o):
            return o.item()
        if LazyType["ext.PdDataFrame"]("pandas.DataFrame").isinstance(o):
            return o.to_dict()  # type: ignore
        if LazyType["ext.PdSeries"]("pandas.Series").isinstance(o):
            return o.to_dict()  # type: ignore
        if LazyType["pydantic.BaseModel"]("pydantic.BaseModel").isinstance(o):
            obj_dict = o.dict()
            if "__root__" in obj_dict:
                obj_dict = obj_dict.get("__root__")
            return obj_dict

        return super().default(o)


class JSON(IODescriptor[JSONType]):
    """
    :code:`JSON` defines API specification for the inputs/outputs of a Service, where either
    inputs will be converted to or outputs will be converted from a JSON representation
    as specified in your API function signature.

    Sample implementation of a sklearn service:

    .. code-block:: python

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

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "iris-classifier" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

        .. code-tab:: python

            import requests
            requests.post(
                "http://0.0.0.0:3000/predict",
                headers={"content-type": "application/json"},
                data='[{"0":5,"1":4,"2":3,"3":2}]'
            ).text

        .. code-tab:: bash

            % curl -X POST -H "Content-Type: application/json" --data '[{"0":5,"1":4,"2":3,"3":2}]' http://0.0.0.0:3000/predict

            {"res":"[{\"0\":1}]"}%

    Args:
        pydantic_model (:code:`pydantic.BaseModel`, `optional`, default to :code:`None`):
            Pydantic model schema.
        validate_json (:code:`bool`, `optional`, default to :code:`True`): If True, then use
            Pydantic model specified above to validate given JSON.
        json_encoder (:code:`Type[json.JSONEncoder]`, default to :code:`~bentoml._internal.io_descriptor.json.DefaultJsonEncoder`):
            JSON encoder class.

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that in JSON format.
    """

    def __init__(
        self,
        pydantic_model: t.Type[pydantic.BaseModel] | None = None,
        validate_json: bool = True,
        json_encoder: t.Type[json.JSONEncoder] = DefaultJsonEncoder,
    ):
        if pydantic_model is not None:
            try:
                import pydantic
            except ImportError:
                raise MissingDependencyException(
                    "`pydantic` must be installed to use `pydantic_model`"
                )
            assert issubclass(
                pydantic_model, pydantic.BaseModel
            ), "`pydantic_model` must be a subclass of `pydantic.BaseModel`"

        self._pydantic_model = pydantic_model
        self._validate_json = validate_json
        self._json_encoder = json_encoder

    def input_type(self) -> "UnionType":
        return JSONType

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        if self._pydantic_model is None:
            return {"type": "object"}
        return self._pydantic_model.schema()

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    async def from_http_request(self, request: Request) -> JSONType:
        json_str = await request.body()
        if self._pydantic_model is not None and self._validate_json:
            try:
                import pydantic
            except ImportError:
                raise MissingDependencyException(
                    "`pydantic` must be installed to use `pydantic_model`"
                ) from None
            try:
                pydantic_model = self._pydantic_model.parse_raw(json_str)
                return pydantic_model.dict()
            except pydantic.ValidationError as e:
                raise BadInput(f"Json validation error: {e}") from None
        else:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise BadInput(f"Json validation error: {e}") from None

    async def to_http_response(self, obj: JSONType, ctx: Context | None = None):
        json_str = json.dumps(
            obj,
            cls=self._json_encoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        )

        if ctx is not None:
            res = Response(
                json_str,
                media_type=MIME_TYPE_JSON,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(json_str, media_type=MIME_TYPE_JSON)
