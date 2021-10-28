import json
import typing as t

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor


class Text(IODescriptor):
    """

    `Text` defines API specification for the inputs/outputs of a Service. `Text` represents strings
      for all incoming requests/outcoming responses as specified in your API function signature.

    .. Toy implementation of a GPT2 service::
        # gpt2_svc.py
        import bentoml
        from bentoml.io import Text
        import bentoml.transformers

        # If you don't have a gpt2 model previously saved under BentoML modelstore
        # tag = bentoml.transformers.import_from_huggingface_hub('gpt2')
        runner = bentoml.transformers.load_runner('gpt2',tasks='text-generation')

        svc = bentoml.Service("server", runners=[runner])

        @svc.api(input=Text(), output=Text())
        def predict(input_arr):
            res = runner.run_batch(input_arr)
            return {"results": res}

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./gpt2_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "IrisClassifierService" defined in "gpt2_svc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        % curl -X POST -H "Content-Type: text/plain" --data 'Not for nothing did Orin say that people outdoors down here just scuttle in vectors from air coniditioning to air conditioning.' http://0.0.0.0:5000/predict

        {"results": [{"generated_text": "Not for nothing did Orin say that people outdoors down here just scuttle in vectors from air coniditioning to air conditioning.\n\nA few years ago, when I traveled to Mexico, I came across a local artist's piece called"}]}%

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

    @t.overload
    async def to_http_response(self, obj: str) -> Response:
        ...

    @t.overload
    async def to_http_response(self, obj: t.Dict[str, str]) -> Response:  # noqa: F811
        ...

    async def to_http_response(  # noqa: F811
        self, obj: t.Union[str, t.Dict[str, t.Any]]
    ) -> Response:
        if isinstance(obj, str):
            MIME_TYPE = "text/plain"
            resp = obj
        else:
            MIME_TYPE = "application/json"
            resp = json.dumps(obj)
        return Response(resp, media_type=MIME_TYPE)
