from __future__ import annotations

import json
import typing as t
import logging
import dataclasses
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

import attr
from starlette.requests import Request
from starlette.responses import Response

from bentoml.exceptions import BadInput
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import UnprocessableEntity

from .base import IODescriptor
from ..types import LazyType
from ..utils import LazyLoader
from ..utils import bentoml_cattr
from ..utils.http import set_cookies
from ..service.openapi import REF_PREFIX
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

if TYPE_CHECKING:
    from types import UnionType

    import numpy as np
    import pydantic
    import pydantic.schema as schema

    from bentoml.grpc.v1 import service_pb2 as pb

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context
    from ..server.grpc.types import BentoServicerContext

    class RawPayload(t.TypedDict, total=False):
        kind: str
        metadata: t.Dict[str, str]
        content: bytes

    class ArrayPayload(t.TypedDict, total=False):
        bool_values: t.List[bool]
        float_values: t.List[float]
        string_values: t.List[str]
        double_values: t.List[float]
        int_values: t.List[int]
        long_values: t.List[int]
        uint32_values: t.List[int]
        uint64_values: t.List[int]

    class NDArrayPayload(t.TypedDict, total=False):
        shape: int
        array: ArrayPayload

    class MapPayload(t.TypedDict):
        fields: dict[str, ValuePayload]

    class ValuePayload(t.TypedDict, total=False):
        string_value: str
        raw_value: RawPayload
        array_value: ArrayPayload
        map_value: MapPayload

else:
    _exc_msg = "'pydantic' must be installed to use 'pydantic_model'. Install with 'pip install pydantic'."
    pydantic = LazyLoader("pydantic", globals(), "pydantic", exc_msg=_exc_msg)
    schema = LazyLoader("schema", globals(), "pydantic.schema", exc_msg=_exc_msg)

    # lazy load our proto generated.
    pb = LazyLoader("pb", globals(), "bentoml.grpc.v1.service_pb2")
    # lazy load numpy for processing ndarray.
    np = LazyLoader("np", globals(), "numpy")


ListT = t.List[t.Union[int, bool, float, str]]

JSONType = t.Union[str, t.Dict[str, t.Any], "pydantic.BaseModel", None]

MIME_TYPE_JSON = "application/json"

logger = logging.getLogger(__name__)


class DefaultJsonEncoder(json.JSONEncoder):
    def default(self, o: type) -> t.Any:
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
        if attr.has(o):
            return bentoml_cattr.unstructure(o)

        return super().default(o)

    multi_dimensional_array_value: NDArrayPayload


class JSON(IODescriptor[JSONType], proto_fields=["map_value", "raw_value"]):
    """
    :obj:`JSON` defines API specification for the inputs/outputs of a Service, where either
    inputs will be converted to or outputs will be converted from a JSON representation
    as specified in your API function signature.

    A sample service implementation:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       import typing
       from typing import TYPE_CHECKING
       from typing import Any
       from typing import Optional

       import bentoml
       from bentoml.io import NumpyNdarray
       from bentoml.io import JSON

       import numpy as np
       import pandas as pd
       from pydantic import BaseModel

       iris_clf_runner = bentoml.sklearn.get("iris_clf_with_feature_names:latest").to_runner()

       svc = bentoml.Service("iris_classifier_pydantic", runners=[iris_clf_runner])

       class IrisFeatures(BaseModel):
           sepal_len: float
           sepal_width: float
           petal_len: float
           petal_width: float

           # Optional field
           request_id: Optional[int]

           # Use custom Pydantic config for additional validation options
           class Config:
               extra = 'forbid'


       input_spec = JSON(pydantic_model=IrisFeatures)

       @svc.api(input=input_spec, output=NumpyNdarray())
       def classify(input_data: IrisFeatures) -> NDArray[Any]:
           if input_data.request_id is not None:
               print("Received request ID: ", input_data.request_id)

           input_df = pd.DataFrame([input_data.dict(exclude={"request_id"})])
           return iris_clf_runner.run(input_df)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

               % curl -X POST -H "content-type: application/json" \\
                   --data '{"sepal_len": 6.2, "sepal_width": 3.2, "petal_len": 5.2, "petal_width": 2.2}' \\
                   http://127.0.0.1:3000/classify

               # [2]%

        .. tab-item:: Python

            .. code-block:: python
               :caption: `request.py`

                import requests

                requests.post(
                    "http://0.0.0.0:3000/predict",
                    headers={"content-type": "application/json"},
                    data='{"sepal_len": 6.2, "sepal_width": 3.2, "petal_len": 5.2, "petal_width": 2.2}'
                ).text

    Args:
        pydantic_model: Pydantic model schema. When used, inference API callback
                        will receive an instance of the specified ``pydantic_model`` class.
        json_encoder: JSON encoder class. By default BentoML implements a custom JSON encoder that
                      provides additional serialization supports for numpy arrays, pandas dataframes,
                      dataclass-like (`attrs <https://www.attrs.org/en/stable/>`_, dataclass, etc.).
                      If you wish to use a custom encoder, make sure to support the aforementioned object.

    Returns:
        :obj:`JSON`: IO Descriptor that represents JSON format.
    """

    _mime_type: str = MIME_TYPE_JSON

    def __init__(
        self,
        *,
        pydantic_model: t.Type[pydantic.BaseModel] | None = None,
        validate_json: bool | None = None,
        json_encoder: t.Type[json.JSONEncoder] = DefaultJsonEncoder,
        packed: bool = False,
        _chunk_workers: int = 10,
    ):
        if pydantic_model:
            assert issubclass(
                pydantic_model, pydantic.BaseModel
            ), "'pydantic_model' must be a subclass of 'pydantic.BaseModel'."

        self._pydantic_model = pydantic_model
        self._json_encoder = json_encoder

        self._packed = packed
        self._chunk_workers = _chunk_workers

        # Remove validate_json in version 1.0.2
        if validate_json is not None:
            logger.warning(
                "'validate_json' option from 'bentoml.io.JSON' has been deprecated. Use a Pydantic model to specify validation options instead."
            )

    def input_type(self) -> UnionType:
        return JSONType

    def openapi_schema(self) -> Schema:
        if not self._pydantic_model:
            return Schema(type="object")

        # returns schemas from pydantic_model.
        return Schema(
            **schema.model_process_schema(
                self._pydantic_model,
                model_name_map=schema.get_model_name_map(
                    schema.get_flat_models_from_model(self._pydantic_model)
                ),
                ref_prefix=REF_PREFIX,
            )[0]
        )

    def openapi_components(self) -> dict[str, t.Any] | None:
        if not self._pydantic_model:
            return {}

        from ..service.openapi.utils import pydantic_components_schema

        return {"schemas": pydantic_components_schema(self._pydantic_model)}

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
            required=True,
        )

    def openapi_responses(self) -> OpenAPIResponse:
        return OpenAPIResponse(
            description=SUCCESS_DESCRIPTION,
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
        )

    async def from_http_request(
        self, request: Request
    ) -> JSONType | pydantic.BaseModel:
        json_str = await request.body()
        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise BadInput(f"Invalid JSON input received: {e}") from e

        if self._pydantic_model:
            try:
                pydantic_model = self._pydantic_model.parse_obj(json_obj)
                return pydantic_model
            except pydantic.ValidationError as e:
                raise BadInput(f"Invalid JSON input received: {e}") from e
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

    def generate_protobuf(self):
        pass

    async def from_grpc_request(
        self, request: pb.Request, context: BentoServicerContext
    ) -> JSONType | pydantic.BaseModel:
        from ..utils.grpc import check_field
        from ..utils.grpc import deserialize_proto
        from ..utils.grpc import raise_grpc_exception

        field = check_field(request, self)

        if self._packed:
            if field != "raw_value":
                raise_grpc_exception(
                    f"'packed={self._packed}' requires to use 'raw_value' instead of {field}.",
                    context=context,
                    exc_cls=UnprocessableEntity,
                )

            raw = request.input.raw_value

            if self._pydantic_model:
                try:
                    return self._pydantic_model.parse_raw(raw.content)
                except pydantic.ValidationError as e:
                    raise_grpc_exception(
                        f"Invalid JSON input received: {e}",
                        context=context,
                        exc_cls=UnprocessableEntity,
                    )
            try:
                return json.loads(raw.content)
            except json.JSONDecodeError as e:
                raise_grpc_exception(
                    f"Invalid JSON input received: {e}",
                    context=context,
                    exc_cls=BadInput,
                )

        if field == "raw_value":
            raise_grpc_exception(
                "'raw_value' requires 'packed=True'.",
                context=context,
                exc_cls=UnprocessableEntity,
            )

        deserialized = deserialize_proto(request)[field]["fields"]
        iterator = ((k, v) for k, v in deserialized.items())
        print([(k, v) for k, v in iterator])

        with ThreadPoolExecutor(max_workers=self._chunk_workers) as executor:
            json_obj = list(executor.map(process_payload, iterator))
            json_obj = {k: v for k, v in json_obj}
        print(json_obj)

        if self._pydantic_model:
            try:
                return self._pydantic_model.parse_obj(json_obj)
            except pydantic.ValidationError as e:
                raise_grpc_exception(
                    f"Invalid JSON input received: {e}",
                    context=context,
                    exc_cls=UnprocessableEntity,
                )

        return json_obj

    async def to_grpc_response(
        self, obj: JSONType | pydantic.BaseModel, context: BentoServicerContext
    ) -> pb.Response:
        from ..utils.grpc import serialize_proto
        from ..utils.grpc import raise_grpc_exception

        json_str = None

        # use pydantic model for validation.
        if self._pydantic_model:
            try:
                self._pydantic_model.parse_obj(obj)
            except pydantic.ValidationError as e:
                raise raise_grpc_exception(
                    f"Invalid JSON input received: {e}",
                    context=context,
                    exc_cls=BadInput,
                )

        if isinstance(obj, pydantic.BaseModel):
            obj = obj.dict()

        if obj:
            json_str = json.dumps(
                obj,
                cls=self._json_encoder,
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            )


def process_payload(iterator: tuple[str, ValuePayload]) -> tuple[str, t.Any]:
    key, payload = iterator

    if len(payload) > 1:
        raise BadInput("Only oneOf 'service_pb2.Value' is allowed.")

    # We include TYPE_CHECKING logic here since
    # 'elif' check already ensure the value to be not None.
    if payload.get("string_value"):
        res = payload.get("string_value")
    elif payload.get("raw_value"):
        raw = payload.get("raw_value")
        if TYPE_CHECKING:
            assert raw
        res = process_raw_payload(raw)
    elif payload.get("array_value"):
        arr = payload.get("array_value")
        if TYPE_CHECKING:
            assert arr
        # we only care about the array itself here.
        res = process_array_payload(arr)[1]
    elif payload.get("multi_dimensional_array_value"):
        ndarr = payload.get("multi_dimensional_array_value")
        if TYPE_CHECKING:
            assert ndarr
        res = process_multi_dimensional_array_payload(ndarr)
    elif payload.get("map_value"):
        mval = payload.get("map_value")
        if TYPE_CHECKING:
            assert mval
        res = process_map_payload(mval)
    else:
        raise BentoMLException("Invalid payload.")
    return key, res


def process_raw_payload(payload: RawPayload):
    ...


def process_map_payload(payload: MapPayload):
    ...


def process_multi_dimensional_array_payload(payload: NDArrayPayload):
    ...


# array_descriptor -> {"float_contents": [1, 2, 3]}
def process_array_payload(array: ArrayPayload) -> tuple[str, ListT]:
    # returns the array contents with whether the result is using bytes.
    accepted_fields = list(pb.Array.DESCRIPTOR.fields_by_name)
    if len(set(array) - set(accepted_fields)) > 0:
        raise UnprocessableEntity("Given array has unsupported fields.")
    if len(array) != 1:
        raise BadInput(
            f"Array contents can only be one of {accepted_fields} as key. Use one of {list(array)} only."
        )

    # since TypedDict returns tuple[str, object], hence the cast.
    return t.cast(t.Tuple[str, ListT], tuple(array.items())[0])
