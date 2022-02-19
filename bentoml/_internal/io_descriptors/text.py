import typing as t

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor

MIME_TYPE = "text/plain"


class Text(IODescriptor[str]):
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
        runner = bentoml.transformers.load_runner('gpt2',tasks='text-generation')

        svc = bentoml.Service("gpt2-generation", runners=[runner])

        @svc.api(input=Text(), output=Text())
        def predict(input_arr):
            res = runner.run_batch(input_arr)
            return res[0]['generated_text']

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

    async def to_http_response(self, obj: str) -> Response:
        return Response(obj, media_type=MIME_TYPE)
