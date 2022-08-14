from __future__ import annotations

import typing as t
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

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.types import BentoServicerContext

    from ..types import LazyType
    from ..context import InferenceApiContext as Context


class Multipart(IODescriptor[t.Any], proto_field="multipart"):
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
                   |   |                                                        |   |
                   |   +----------------+-----------------------+---------------+   |
                   |                    |                       |                   |
                   |                    |                       |                   |
                   |                    |                       |                   |
                   |                    +----+        +---------+                   |
                   |                         |        |                             |
                   |         +---------------v--------v---------+                   |
                   |         |  def predict(arr, annotations):  |                   |
                   |         +----------------------------------+                   |
                   |                                                                |
                   +----------------------------------------------------------------+

    Returns:
        :obj:`Multipart`: IO Descriptor that represents a Multipart request/response.
    """

    def __init__(self, **inputs: IODescriptor[t.Any]):
        if any(isinstance(descriptor, Multipart) for descriptor in inputs.values()):
            raise InvalidArgument(
                "Multipart IO can not contain nested Multipart IO descriptor"
            )
        self._inputs = inputs
        self._mime_type = "multipart/form-data"

    def input_type(
        self,
    ) -> dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]:
        res: dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]] = {}
        for k, v in self._inputs.items():
            inp_type = v.input_type()
            if isinstance(inp_type, dict):
                raise TypeError(
                    "A multipart descriptor cannot take a multi-valued I/O descriptor as input"
                )
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
            )

        reqs = await populate_multipart_requests(request)

        return {
            key: await io_.from_http_request(reqs[key])
            for key, io_ in self._inputs.items()
        }

    async def to_http_response(
        self, obj: dict[str, t.Any], ctx: Context | None = None
    ) -> Response:
        res_mapping: dict[str, Response] = {
            key: await io_.to_http_response(obj[key], ctx)
            for key, io_ in self._inputs.items()
        }
        return await concat_to_multipart_response(res_mapping, ctx)

    async def from_grpc_request(
        self, request: pb.Request, context: BentoServicerContext
    ) -> dict[str, t.Any]:
        from bentoml.grpc.utils import get_field
        from bentoml.grpc.utils import raise_grpc_exception
        from bentoml.grpc.utils import validate_content_type

        # validate gRPC content type if content type is specified
        validate_content_type(context, self)
        field = get_field(request, self)

        if len(set(field) - set(self._inputs)) != 0:
            raise_grpc_exception(
                f"'{self.__class__.__name__}' only accepts '{set(self._inputs)}' as input fields. Invalid fields are: {set(field) - set(self._inputs)}",
                context=context,
            )

        return {
            key: await self._inputs[key].from_grpc_request(input_pb, context)
            for key, input_pb in field.items()
        }

    async def to_grpc_response(
        self, obj: dict[str, t.Any], context: BentoServicerContext
    ) -> pb.Response:
        from bentoml.grpc.utils import raise_grpc_exception
        from bentoml.grpc.utils import validate_content_type

        # validate gRPC content type if content type is specified
        validate_content_type(context, self)

        context.set_trailing_metadata((("content-type", self.grpc_content_type),))

        if len(set(obj) - set(self._inputs)) != 0:
            raise_grpc_exception(
                f"'{self.__class__.__name__}' only accepts '{set(self._inputs)}' as output fields. Invalid fields are: {set(obj) - set(self._inputs)}",
                context=context,
                exception_cls=InvalidArgument,
            )
        multipart_map: dict[str, pb.Part] = {}

        for key in obj:
            io_descriptor = self._inputs[key]
            resp = await io_descriptor.to_grpc_response(obj[key], context)
            part = pb.Part(
                **{io_descriptor.proto_field: getattr(resp, io_descriptor.proto_field)}
            )
            multipart_map[key] = part

        return pb.Response(multipart=multipart_map)
