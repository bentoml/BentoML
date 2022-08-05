from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from ..utils.http import set_cookies
from ..utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    from bentoml.grpc.v1 import service_pb2 as _service_pb2

    from ..context import InferenceApiContext as Context
    from ...server.grpc.types import BentoServicerContext
else:
    _service_pb2 = LazyLoader("_service_pb2", globals(), "bentoml.grpc.v1.service_pb2")


MIME_TYPE = "text/plain"


class Text(IODescriptor[str], proto_fields=["string_value", "raw_value"]):
    """
    :code:`Text` defines API specification for the inputs/outputs of a Service. :code:`Text`
    represents strings for all incoming requests/outcoming responses as specified in
    your API function signature.

    Sample implementation of a GPT2 service:

    .. code-block:: python

        # gpt2_svc.py
        import bentoml
        from bentoml.io import Text
        import bentoml.transformers

        # If you don't have a gpt2 model previously saved under BentoML modelstore
        # tag = bentoml.transformers.import_from_huggingface_hub('gpt2')
        runner = bentoml.tensorflow.get('gpt2').to_runner()

        svc = bentoml.Service("gpt2-generation", runners=[runner])

        @svc.api(input=Text(), output=Text())
        def predict(input_arr):
            res = runner.run(input_arr)
            return res['generated_text']

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./gpt2_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "gpt2-generation" defined in "gpt2_svc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

        .. code-block:: python

            import requests
            requests.post(
                "http://0.0.0.0:3000/predict",
                headers = {"content-type":"text/plain"},
                data = 'Not for nothing did Orin say that people outdoors down here just scuttle in vectors from air conditioning to air conditioning.'
            ).text

        .. code-block:: bash

            % curl -X POST -H "Content-Type: text/plain" --data 'Not for nothing did Orin
            say that people outdoors down here just scuttle in vectors from air
            conditioning to air conditioning.' http://0.0.0.0:3000/predict

    .. note::

        `Text` is not designed to take any `args` or `kwargs` during initialization

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that strings type.
    """

    def input_type(self) -> t.Type[str]:
        return str

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        return {"type": "string"}

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {MIME_TYPE: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {MIME_TYPE: {"schema": self.openapi_schema_type()}}

    async def from_http_request(self, request: Request) -> str:
        obj = await request.body()
        return str(obj.decode("utf-8"))

    async def to_http_response(self, obj: str, ctx: Context | None = None) -> Response:
        if ctx is not None:
            res = Response(
                obj,
                media_type=MIME_TYPE,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(obj, media_type=MIME_TYPE)

    async def from_grpc_request(
        self,
        request: _service_pb2.Request,
        context: BentoServicerContext,  # pylint: disable=unused-argument
    ) -> str:
        return str(request.input.string_value)

    async def to_grpc_response(
        self, obj: str, context: BentoServicerContext  # pylint: disable=unused-argument
    ) -> _service_pb2.Response:
        return _service_pb2.Response(output=_service_pb2.Value(string_value=obj))

    def generate_protobuf(self):
        pass
