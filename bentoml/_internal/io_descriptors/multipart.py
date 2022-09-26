from __future__ import annotations

import typing as t
import asyncio
from typing import TYPE_CHECKING

from starlette.requests import Request
from multipart.multipart import parse_options_header
from starlette.responses import Response

from .base import IODescriptor
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ..service.openapi import SUCCESS_DESCRIPTION
from ..utils.formparser import populate_multipart_requests
from ..utils.formparser import concat_to_multipart_response
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

if TYPE_CHECKING:
    from types import UnionType

    from bentoml.grpc.v1alpha1 import service_pb2 as pb

    from ..types import LazyType
    from ..context import InferenceApiContext as Context
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()


class Multipart(IODescriptor[t.Dict[str, t.Any]]):
    """
    :obj:`Multipart` defines API specification for the inputs/outputs of a Service, where inputs/outputs
    of a Service can receive/send a **multipart** request/responses as specified in your API function signature.

    A sample service implementation:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       from typing import TYPE_CHECKING
       from typing import Any

       import bentoml
       from bentoml.io import NumpyNdarray
       from bentoml.io import Multipart
       from bentoml.io import JSON

       if TYPE_CHECKING:
           from numpy.typing import NDArray

       runner = bentoml.sklearn.get("sklearn_model_clf").to_runner()

       svc = bentoml.Service("iris-classifier", runners=[runner])

       input_spec = Multipart(arr=NumpyNdarray(), annotations=JSON())
       output_spec = Multipart(output=NumpyNdarray(), result=JSON())

       @svc.api(input=input_spec, output=output_spec)
       async def predict(
           arr: NDArray[Any], annotations: dict[str, Any]
       ) -> dict[str, NDArray[Any] | dict[str, Any]]:
           res = await runner.run(arr)
           return {"output": res, "result": annotations}

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

               % curl -X POST -H "Content-Type: multipart/form-data" \\
                      -F annotations=@test.json -F arr='[5,4,3,2]' \\
                      http://0.0.0.0:3000/predict

               # --b1d72c201a064ecd92a17a412eb9208e
               # Content-Disposition: form-data; name="output"
               # content-length: 1
               # content-type: application/json

               # 1
               # --b1d72c201a064ecd92a17a412eb9208e
               # Content-Disposition: form-data; name="result"
               # content-length: 13
               # content-type: application/json

               # {"foo":"bar"}
               # --b1d72c201a064ecd92a17a412eb9208e--

        .. tab-item:: Python

           .. note::

              The following code snippet uses `requests_toolbelt <https://github.com/requests/toolbelt>`_.
              Install with ``pip install requests-toolbelt``.

           .. code-block:: python
              :caption: `request.py`

              import requests

              from requests_toolbelt.multipart.encoder import MultipartEncoder

              m = MultipartEncoder(
                  fields={
                      "field0": "value",
                      "field1": "value",
                      "field2": ("filename", open("test.json", "rb"), "application/json"),
                  }
              )

              requests.post(
                  "http://0.0.0.0:3000/predict", data=m, headers={"Content-Type": m.content_type}
              )

    Args:
        inputs: Dictionary consisting keys as inputs definition for a Multipart
                request/response, values as IODescriptor supported by BentoML. Currently,
                :obj:`Multipart` supports :obj:`Image`, :obj:`NumpyNdarray`, :obj:`PandasDataFrame`, :obj:`PandasSeries`, :obj:`Text`,
                and :obj:`File`.

                Make sure to match the input parameters in function signatures in an API function to the keys defined
                under :obj:`Multipart`:

                .. code-block:: bash

                   +----------------------------------------------------------------+
                   |                                                                |
                   |   +--------------------------------------------------------+   |
                   |   |                                                        |   |
                   |   |    Multipart(arr=NumpyNdarray(), annotations=JSON())   |   |
                   |   |               |                       |                |   |
                   |   +---------------+-----------------------+----------------+   |
                   |                   |                       |                    |
                   |                   |                       |                    |
                   |                   |                       |                    |
                   |                   +-----+        +--------+                    |
                   |                         |        |                             |
                   |         +---------------v--------v---------+                   |
                   |         |  def predict(arr, annotations):  |                   |
                   |         +----------------------------------+                   |
                   |                                                                |
                   +----------------------------------------------------------------+

    Returns:
        :obj:`Multipart`: IO Descriptor that represents a Multipart request/response.
    """

    _proto_fields = ("multipart",)
    _mime_type = "multipart/form-data"

    def __init__(self, **inputs: IODescriptor[t.Any]):
        if any(isinstance(descriptor, Multipart) for descriptor in inputs.values()):
            raise InvalidArgument(
                "Multipart IO can not contain nested Multipart IO descriptor"
            ) from None
        self._inputs = inputs

    def __repr__(self) -> str:
        return f"Multipart({','.join([f'{k}={v}' for k,v in zip(self._inputs, map(repr, self._inputs.values()))])})"

    def input_type(
        self,
    ) -> dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]:
        res: dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]] = {}
        for k, v in self._inputs.items():
            inp_type = v.input_type()
            if isinstance(inp_type, dict):
                raise TypeError(
                    "A multipart descriptor cannot take a multi-valued I/O descriptor as input"
                ) from None
            res[k] = inp_type

        return res

    def openapi_schema(self) -> Schema:
        return Schema(
            type="object",
            properties={args: io.openapi_schema() for args, io in self._inputs.items()},
        )

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

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

    async def from_http_request(self, request: Request) -> dict[str, t.Any]:
        ctype, _ = parse_options_header(request.headers["content-type"])
        if ctype != b"multipart/form-data":
            raise BentoMLException(
                f"{self.__class__.__name__} only accepts `multipart/form-data` as Content-Type header, got {ctype} instead."
            ) from None

        to_populate = zip(
            self._inputs.values(), (await populate_multipart_requests(request)).values()
        )
        reqs = await asyncio.gather(
            *tuple(io_.from_http_request(req) for io_, req in to_populate)
        )
        return dict(zip(self._inputs, reqs))

    async def to_http_response(
        self, obj: dict[str, t.Any], ctx: Context | None = None
    ) -> Response:
        resps = await asyncio.gather(
            *tuple(
                io_.to_http_response(obj[key], ctx) for key, io_ in self._inputs.items()
            )
        )
        return await concat_to_multipart_response(dict(zip(self._inputs, resps)), ctx)

    def validate_input_mapping(self, field: t.MutableMapping[str, t.Any]) -> None:
        if len(set(field) - set(self._inputs)) != 0:
            raise InvalidArgument(
                f"'{self!r}' accepts the following keys: {set(self._inputs)}. Given {field.__class__.__qualname__} has invalid fields: {set(field) - set(self._inputs)}",
            ) from None

    async def from_proto(self, field: pb.Multipart) -> dict[str, t.Any]:
        from bentoml.grpc.utils import validate_proto_fields

        if isinstance(field, bytes):
            raise InvalidArgument(
                f"cannot use 'serialized_bytes' with {self.__class__.__name__}"
            ) from None
        message = field.fields
        self.validate_input_mapping(message)
        to_populate = {self._inputs[k]: message[k] for k in self._inputs}
        reqs = await asyncio.gather(
            *tuple(
                descriptor.from_proto(
                    getattr(
                        part,
                        validate_proto_fields(
                            part.WhichOneof("representation"), descriptor
                        ),
                    )
                )
                for descriptor, part in to_populate.items()
            )
        )
        return dict(zip(self._inputs.keys(), reqs))

    async def to_proto(self, obj: dict[str, t.Any]) -> pb.Multipart:
        self.validate_input_mapping(obj)
        resps = await asyncio.gather(
            *tuple(
                io_.to_proto(data)
                for io_, data in zip(self._inputs.values(), obj.values())
            )
        )
        return pb.Multipart(
            fields=dict(
                zip(
                    obj,
                    [
                        # TODO: support multiple proto_fields
                        pb.Part(**{io_._proto_fields[0]: resp})
                        for io_, resp in zip(self._inputs.values(), resps)
                    ],
                )
            )
        )
