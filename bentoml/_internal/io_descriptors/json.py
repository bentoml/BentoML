from __future__ import annotations

import json
import typing as t
import logging
import dataclasses
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import Parameter
from ..service.openapi.specification import Reference
from ..service.openapi.specification import Components
from ..service.openapi.specification import RequestBody

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
else:
    pydantic = LazyLoader(
        "pydantic",
        globals(),
        "pydantic",
        exc_msg="`pydantic` must be installed to use `pydantic_model`, install with `pip install pydantic`",
    )


JSONType = t.Union[str, t.Dict[str, t.Any], "pydantic.BaseModel", None]

MIME_TYPE_JSON = "application/json"

logger = logging.getLogger(__name__)


class DefaultJsonEncoder(json.JSONEncoder):
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

        import typing
        import numpy as np
        import pandas as pd
        import bentoml
        from bentoml.io import NumpyNdarray, JSON
        from pydantic import BaseModel

        iris_clf_runner = bentoml.sklearn.get("iris_clf_with_feature_names:latest").to_runner()

        svc = bentoml.Service("iris_classifier_pydantic", runners=[iris_clf_runner])

        class IrisFeatures(BaseModel):
            sepal_len: float
            sepal_width: float
            petal_len: float
            petal_width: float

            # Optional field
            request_id: typing.Optional[int]

            # Use custom Pydantic config for additional validation options
            class Config:
                extra = 'forbid'


        input_spec = JSON(pydantic_model=IrisFeatures)

        @svc.api(input=input_spec, output=NumpyNdarray())
        def classify(input_data: IrisFeatures) -> np.ndarray:
            if input_data.request_id is not None:
                print("Received request ID: ", input_data.request_id)

            input_df = pd.DataFrame([input_data.dict(exclude={"request_id"})])
            return iris_clf_runner.predict.run(input_df)


    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc

        [INFO] [cli] Starting development BentoServer from "service.py:svc" running on http://127.0.0.1:3000 (Press CTRL+C to quit)

    Users can then send requests to the newly started services with any client:

        % curl -X POST -H "content-type: application/json" \
            --data '{"sepal_len": 6.2, "sepal_width": 3.2, "petal_len": 5.2, "petal_width": 2.2}' \
            http://127.0.0.1:3000/classify

        [2]%

    Args:
        pydantic_model (:code:`pydantic.BaseModel`, `optional`, default to :code:`None`):
            Pydantic model schema. When used, inference API callback will receive an instance of the specified pydantic_model class
        json_encoder (:code:`Type[json.JSONEncoder]`, default to :code:`~bentoml._internal.io_descriptor.json.DefaultJsonEncoder`):
            JSON encoder class.

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that in JSON format.
    """

    _mime_type: str = MIME_TYPE_JSON

    def __init__(
        self,
        *,
        pydantic_model: t.Type[pydantic.BaseModel] | None = None,
        validate_json: bool | None = None,
        json_encoder: t.Type[json.JSONEncoder] = DefaultJsonEncoder,
    ):
        if pydantic_model:
            assert issubclass(
                pydantic_model, pydantic.BaseModel
            ), "'pydantic_model' must be a subclass of 'pydantic.BaseModel'."

        self._pydantic_model = pydantic_model
        self._json_encoder = json_encoder

        # Remove validate_json in version 1.0.2
        if validate_json is not None:
            logger.warning(
                "'validate_json' option from 'bentoml.io.JSON' has been deprecated. Use a Pydantic model to specify validation options instead."
            )

    def input_type(self) -> UnionType:
        return JSONType

    def openapi_schema(self) -> Schema | Reference:
        if not self._pydantic_model:
            return Schema(type="object")
        from ..service.openapi.utils import generate_model_schema

        generated = generate_model_schema(self._pydantic_model)
        return

    def openapi_parameter(self) -> Parameter | Reference:
        pass

    def openapi_components(self) -> Components:
        pass

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={self._mime_type: MediaType(schema=self.openapi_schema())}
        )

    def openapi_responses(self) -> OpenAPIResponse:
        pass

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        if self._pydantic_model is None:
            return {"type": "object"}

        from ..service.openapi.utils import generate_model_schema

        print(generate_model_schema(self._pydantic_model))

        return self._pydantic_model.schema()

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {MIME_TYPE_JSON: MediaType(schema=self.openapi_schema_type())}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {MIME_TYPE_JSON: MediaType(schema=self.openapi_schema_type())}

    async def from_http_request(self, request: Request) -> JSONType:
        json_str = await request.body()
        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise BadInput(f"Invalid JSON input received: {e}") from None

        if self._pydantic_model is not None:
            try:
                pydantic_model = self._pydantic_model.parse_obj(json_obj)
                return pydantic_model
            except pydantic.ValidationError as e:
                raise BadInput(f"Invalid JSON input received: {e}") from None
        else:
            return json_obj

    async def to_http_response(
        self, obj: JSONType | pydantic.BaseModel, ctx: Context | None = None
    ):
        if isinstance(obj, pydantic.BaseModel):
            obj = obj.dict()

        json_str = (
            json.dumps(
                obj,
                cls=self._json_encoder,
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            )
            if obj is not None
            else None
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
