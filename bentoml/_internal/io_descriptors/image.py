import io
import typing as t

from starlette.requests import Request
from starlette.responses import StreamingResponse

import bentoml._internal.constants as const
from bentoml.exceptions import InvalidArgument

from ..utils import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import numpy as np
    import PIL
    import PIL.Image
else:
    np = LazyLoader("np", globals(), "numpy")

    # NOTE: pillow-simd only benefits users who want to do preprocessing
    # TODO: add options for users to choose between simd and native mode
    _exc = const.IMPORT_ERROR_MSG.format(
        fwr="Pillow",
        module=f"{__name__}.Image",
        inst="`pip install Pillow`",
    )
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=_exc)
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image", exc_msg=_exc)

DEFAULT_PIL_MODE = "RGB"


class Image(IODescriptor):
    """
    `Image` defines API specification for the inputs/outputs of a Service, where either inputs will be
    converted to or outputs will be converted from type `numpy.ndarray` as specified in your API function signature.

    .. Toy implementation of a transformers service for object detection::
        #obj_detc.py
        import bentoml
        from bentoml.io import Image
        import bentoml.transformers

        runner = bentoml.transformers.load_runner("vit", tasks='object-detection')

        svc = bentoml.Service("vit-object-detection", runner=[runner])

        @svc.api(input=Image(), output=Image())
        def predict(input_img):
            return runner(input_img)

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./obj_detc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "vit-object-detection" defined in "obj_detc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        # we will run on our input image test.png
        % curl -X POST -F image=@test.png http://0.0.0.0:5000/predict

        [{"0": 1}]%

    Args:
        pilmode (`str`, `optional`, default to `RGB`):
            Color mode for PIL.
    Returns:
        IO Descriptor that represents either a `np.ndarray` or a `PIL.Image.Image`.
    """

    def __init__(self, pilmode=DEFAULT_PIL_MODE):
        self._pilmode = pilmode

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

    async def from_http_request(self, request: Request) -> "PIL.Image":
        form = await request.form()
        contents = await form["image"].read()
        return PIL.Image.open(io.BytesIO(contents))

    async def to_http_response(
        self, obj: t.Union["np.ndarray", "PIL.Image.Image"]
    ) -> StreamingResponse:
        if not any(isinstance(obj, i) for i in [np.ndarray, PIL.Image.Image]):
            raise InvalidArgument(
                f"Unsupported Image type received: {type(obj)},"
                " `bentoml.io.Image` supports only `np.ndarray` and `PIL.Image`"
            )
        image = PIL.Image.fromarray(obj) if isinstance(obj, np.ndarray) else obj

        # TODO: Support other return types?
        ret = io.BytesIO()
        image.save(ret, "PNG")
        return StreamingResponse(ret, media_type="image/png")
