from __future__ import annotations

import io
import typing as t
from typing import TYPE_CHECKING
from urllib.parse import quote

from starlette.requests import Request
from multipart.multipart import parse_options_header
from starlette.responses import Response

from .base import IODescriptor
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import InvalidArgument
from ...exceptions import InternalServerError

if TYPE_CHECKING:
    from types import UnionType

    import PIL.Image

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context

    _Mode = t.Literal[
        "1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"
    ]
else:

    # NOTE: pillow-simd only benefits users who want to do preprocessing
    # TODO: add options for users to choose between simd and native mode
    _exc = f"""\
    `Pillow` is required to use {__name__}
    Instructions: `pip install -U Pillow`
    """
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=_exc)
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image", exc_msg=_exc)

# NOTES: we will keep type in quotation to avoid backward compatibility
#  with numpy < 1.20, since we will use the latest stubs from the main branch of numpy.
#  that enable a new way to type hint an ndarray.
ImageType: t.TypeAlias = t.Union["PIL.Image.Image", "ext.NpNDArray"]

DEFAULT_PIL_MODE = "RGB"


class Image(IODescriptor[ImageType]):
    """
    :code:`Image` defines API specification for the inputs/outputs of a Service, where either
    inputs will be converted to or outputs will be converted from images as specified
    in your API function signature.

    Sample implementation of a transformers service for object detection:

    .. code-block:: python

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

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./obj_detc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "vit-object-detection" defined in "obj_detc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

        .. code-tab:: python

            import requests
            requests.post(
                "http://0.0.0.0:3000/predict",
                files = {"upload_file": open('test.jpg', 'rb')},
                headers = {"content-type": "multipart/form-data"}
            ).text

        .. code-tab:: bash

            # we will run on our input image test.png
            # image can get from http://images.cocodataset.org/val2017/000000039769.jpg
            % curl -H "Content-Type: multipart/form-data" -F 'fileobj=@test.jpg;type=image/jpeg' http://0.0.0.0:3000/predict

            [{"score":0.8610631227493286,"label":"Egyptian cat"},
            {"score":0.08770329505205154,"label":"tabby, tabby cat"},
            {"score":0.03540956228971481,"label":"tiger cat"},
            {"score":0.004140055272728205,"label":"lynx, catamount"},
            {"score":0.0009498853469267488,"label":"Siamese cat, Siamese"}]%

    Args:
        pilmode (:code:`str`, `optional`, default to :code:`RGB`):
            Color mode for PIL.
        mime_type (:code:`str`, `optional`, default to :code:`image/jpeg`):
            Return MIME type of the :code:`starlette.response.Response`, only available
            when used as output descriptor.

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that either a :code:`PIL.Image.Image` or a :code:`np.ndarray`
        representing an image.

    """  # noqa

    MIME_EXT_MAPPING: t.Dict[str, str] = {}

    def __init__(
        self,
        pilmode: _Mode | None = DEFAULT_PIL_MODE,
        mime_type: str = "image/jpeg",
    ):
        try:
            import PIL.Image
        except ImportError:
            raise InternalServerError(
                "`Pillow` is required to use {__name__}\n Instructions: `pip install -U Pillow`"
            )
        PIL.Image.init()
        self.MIME_EXT_MAPPING.update({v: k for k, v in PIL.Image.MIME.items()})

        if mime_type.lower() not in self.MIME_EXT_MAPPING:  # pragma: no cover
            raise InvalidArgument(
                f"Invalid Image mime_type '{mime_type}', "
                f"Supported mime types are {', '.join(PIL.Image.MIME.values())} "
            )
        if pilmode is not None and pilmode not in PIL.Image.MODES:  # pragma: no cover
            raise InvalidArgument(
                f"Invalid Image pilmode '{pilmode}', "
                f"Supported PIL modes are {', '.join(PIL.Image.MODES)} "
            )

        self._mime_type = mime_type.lower()
        self._pilmode: _Mode | None = pilmode
        self._format = self.MIME_EXT_MAPPING[mime_type]

    def input_type(self) -> UnionType:
        return ImageType

    def openapi_schema_type(self) -> dict[str, str]:
        return {"type": "string", "format": "binary"}

    def openapi_request_schema(self) -> dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    async def from_http_request(self, request: Request) -> ImageType:
        content_type, _ = parse_options_header(request.headers["content-type"])
        mime_type = content_type.decode().lower()
        if mime_type == "multipart/form-data":
            form = await request.form()
            bytes_ = await next(iter(form.values())).read()
        elif mime_type.startswith("image/") or mime_type == self._mime_type:
            bytes_ = await request.body()
        else:
            raise BadInput(
                f"{self.__class__.__name__} should get `multipart/form-data`, "
                f"`{self._mime_type}` or `image/*`, got {content_type} instead"
            )
        return PIL.Image.open(io.BytesIO(bytes_))

    async def init_http_response(self) -> Response:
        return Response(None, media_type=self._mime_type)

    async def to_http_response(
        self, obj: ImageType, ctx: Context | None = None
    ) -> Response:
        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(obj):
            image = PIL.Image.fromarray(obj, mode=self._pilmode)
        elif LazyType[PIL.Image.Image]("PIL.Image.Image").isinstance(obj):
            image = obj
        else:
            raise InternalServerError(
                f"Unsupported Image type received: {type(obj)}, `{self.__class__.__name__}`"
                " only supports `np.ndarray` and `PIL.Image`"
            )
        filename = f"output.{self._format.lower()}"

        ret = io.BytesIO()
        image.save(ret, format=self._format)

        # rfc2183
        content_disposition_filename = quote(filename)
        if content_disposition_filename != filename:
            content_disposition = "attachment; filename*=utf-8''{}".format(
                content_disposition_filename
            )
        else:
            content_disposition = f'attachment; filename="{filename}"'

        if ctx is not None:
            if "content-disposition" not in ctx.response.headers:
                ctx.response.headers["content-disposition"] = content_disposition
            res = Response(
                ret.getvalue(),
                media_type=self._mime_type,
                headers=ctx.response.headers,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(
                ret.getvalue(),
                media_type=self._mime_type,
                headers={"content-disposition": content_disposition},
            )
