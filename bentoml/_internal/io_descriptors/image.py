import io
import mimetypes
import sys
import typing as t

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response

from ...exceptions import BentoMLException, InvalidArgument
from ..utils import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import numpy as np
    import PIL
    import PIL.Image

    from .numpy import NumpyType
else:
    np = LazyLoader("np", globals(), "numpy")

    # NOTE: pillow-simd only benefits users who want to do preprocessing
    # TODO: add options for users to choose between simd and native mode
    _exc = f"""\
    `Pillow` is required to use {__name__}
    Instructions: `pip install -U Pillow`
    """
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=_exc)
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image", exc_msg=_exc)

if sys.version_info >= (3, 8):
    Literal = t.Literal
else:
    from typing_extensions import Literal


DEFAULT_PIL_MODE = "RGB"

_Mode = Literal[
    "1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"
]

ImageType = t.Union["PIL.Image.Image", "NumpyType"]


class Image(IODescriptor[ImageType]):
    """
    `Image` defines API specification for the inputs/outputs of a Service, where either
     inputs will be converted to or outputs will be converted from images as specified
     in your API function signature.

    .. Toy implementation of a transformers service for object detection::
        #obj_detc.py
        import bentoml
        from bentoml.io import Image, JSON
        import bentoml.transformers

        tag='google_vit_large_patch16_224:latest'
        runner = bentoml.transformers.load_runner(tag, tasks='image-classification',
                                                  device=-1,
                                                  feature_extractor="google/vit-large-patch16-224")

        svc = bentoml.Service("vit-object-detection", runners=[runner])

        @svc.api(input=Image(), output=JSON())
        def predict(input_img):
            res = runner.run_batch(input_img)
            return res

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./obj_detc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "vit-object-detection" defined in "obj_detc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        # we will run on our input image test.png
        # image can get from http://images.cocodataset.org/val2017/000000039769.jpg
        % curl -H "Content-Type: multipart/form-data" -F
          'fileobj=@test.jpg;type=image/jpeg' http://0.0.0.0:8000/predict

        [{"score":0.8610631227493286,"label":"Egyptian cat"},
         {"score":0.08770329505205154,"label":"tabby, tabby cat"},
         {"score":0.03540956228971481,"label":"tiger cat"},
         {"score":0.004140055272728205,"label":"lynx, catamount"},
         {"score":0.0009498853469267488,"label":"Siamese cat, Siamese"}]%

    Args:
        pilmode (`str`, `optional`, default to `RGB`):
            Color mode for PIL.
        mime_type (`str`, `optional`, default to `image/jpeg`):
            Return MIME type for `starlette.response.Response`

    Returns:
        IO Descriptor that represents either a `np.ndarray` or a `PIL.Image.Image`.
    """

    def __init__(
        self,
        pilmode: t.Optional["_Mode"] = DEFAULT_PIL_MODE,
        mime_type: str = "image/jpeg",
    ):
        PIL.Image.init()

        # NOTE: Currently no tests are provided.
        if mime_type.lower() not in PIL.Image.MIME.values():  # pragma: no cover
            raise InvalidArgument(
                f"Invalid Image mime_type '{mime_type}', "
                f"Supported mime types are {', '.join(PIL.Image.MIME.values())} "
            )
        if pilmode not in PIL.Image.MODES:  # pragma: no cover
            raise InvalidArgument(
                f"Invalid Image pilmode '{pilmode}', "
                f"Supported PIL modes are {', '.join(PIL.Image.MODES)} "
            )

        self._mime_type = mime_type.lower()
        self._pilmode: t.Optional[_Mode] = pilmode

        ext = mimetypes.guess_extension(self._mime_type)
        if not ext:
            raise BentoMLException("Cannot guess extensions from give mime_type")
        self._format = PIL.Image.EXTENSION[ext]

    def openapi_schema(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        return {self._mime_type: dict(schema=dict(type="string", format="binary"))}

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return self.openapi_schema()

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return self.openapi_schema()

    async def from_http_request(self, request: Request) -> ImageType:
        content_type, _ = parse_options_header(request.headers["content-type"])
        if content_type == b"multipart/form-data":
            form = await request.form()
            contents: bytes = await next(iter(form.values())).read()
            return PIL.Image.open(io.BytesIO(contents))

        if (
            content_type.decode("utf-8").startswith("image/")
            or content_type == self._mime_type
        ):
            return PIL.Image.open(io.BytesIO(await request.body()))

        raise BentoMLException(
            f"{self.__class__.__name__} should have `Content-Type: multipart/form-data` or `image/*`, got {content_type} instead"
        )

    async def to_http_response(self, obj: ImageType) -> Response:
        image = (
            PIL.Image.fromarray(obj, mode=self._pilmode)
            if isinstance(obj, np.ndarray)
            else obj
        )

        ret = io.BytesIO()
        image.save(ret, self._format)
        return Response(ret.getvalue(), media_type=self._mime_type)
