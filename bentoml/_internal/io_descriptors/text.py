import typing as t

from starlette.requests import Request
from starlette.responses import Response

from ...exceptions import InvalidArgument
from .base import IODescriptor


# for review: check output formatting after fixing line breaks
class Text(IODescriptor):
    """

    `Text` defines API specification for the inputs/outputs of a Service. `Text`
      represents strings for all incoming requests/outcoming responses as specified in
      your API function signature.

    .. Toy implementation of a GPT2 service::
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

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./gpt2_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "gpt2-generation" defined in "gpt2_svc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        % curl -X POST -H "Content-Type: text/plain" --data 'Not for nothing did Orin
         say that people outdoors down here just scuttle in vectors from air
         conditioning to air conditioning.' http://0.0.0.0:5000/predict

        Not for nothing did Orin say that people outdoors down here just scuttle in
         vectors from air conditioning to air conditioning. How do you get to such a
         situation?\n\nWell, you want it to just be one giant monster and that is%

    .. notes::
        `Text` is not designed to take any `args` or `kwargs` during initialization

    Returns:
        IO Descriptor that represents strings type.
    """

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

    async def from_http_request(self, request: Request) -> str:
        obj = await request.body()
        return str(obj.decode("utf-8"))

    async def to_http_response(self, obj: str) -> Response:
        if not isinstance(obj, str):
            raise InvalidArgument(
                f"return object is not of type `str`, got type {type(obj)} instead"
            )
        MIME_TYPE = "text/plain"
        resp = obj
        return Response(resp, media_type=MIME_TYPE)
