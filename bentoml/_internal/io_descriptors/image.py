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
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

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
    _exc = f"'Pillow' is required to use {__name__}. Install with: 'pip install -U Pillow'."
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=_exc)
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image", exc_msg=_exc)

# NOTES: we will keep type in quotation to avoid backward compatibility
#  with numpy < 1.20, since we will use the latest stubs from the main branch of numpy.
#  that enable a new way to type hint an ndarray.
ImageType: t.TypeAlias = t.Union["PIL.Image.Image", "ext.NpNDArray"]

DEFAULT_PIL_MODE = "RGB"


class Image(IODescriptor[ImageType]):
    """
    :obj:`Image` defines API specification for the inputs/outputs of a Service, where either
    inputs will be converted to or outputs will be converted from images as specified
    in your API function signature.

    A sample object detection service:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       from typing import TYPE_CHECKING
       from typing import Any

       import bentoml
       from bentoml.io import Image
       from bentoml.io import NumpyNdarray

       if TYPE_CHECKING:
           from PIL.Image import Image
           from numpy.typing import NDArray

       runner = bentoml.tensorflow.get('image-classification:latest').to_runner()

       svc = bentoml.Service("vit-object-detection", runners=[runner])

       @svc.api(input=Image(), output=NumpyNdarray(dtype="float32"))
       async def predict_image(f: Image) -> NDArray[Any]:
           assert isinstance(f, Image)
           arr = np.array(f) / 255.0
           assert arr.shape == (28, 28)

           # We are using greyscale image and our PyTorch model expect one
           # extra channel dimension
           arr = np.expand_dims(arr, (0, 3)).astype("float32")  # reshape to [1, 28, 28, 1]
           return await runner.async_run(arr)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

               # we will run on our input image test.png
               # image can get from http://images.cocodataset.org/val2017/000000039769.jpg
               % curl -H "Content-Type: multipart/form-data" \\
                      -F 'fileobj=@test.jpg;type=image/jpeg' \\
                      http://0.0.0.0:3000/predict_image

               # [{"score":0.8610631227493286,"label":"Egyptian cat"},
               # {"score":0.08770329505205154,"label":"tabby, tabby cat"},
               # {"score":0.03540956228971481,"label":"tiger cat"},
               # {"score":0.004140055272728205,"label":"lynx, catamount"},
               # {"score":0.0009498853469267488,"label":"Siamese cat, Siamese"}]%

        .. tab-item:: Python

           .. code-block:: python
              :caption: `request.py`

              import requests

              requests.post(
                  "http://0.0.0.0:3000/predict_image",
                  files = {"upload_file": open('test.jpg', 'rb')},
                  headers = {"content-type": "multipart/form-data"}
              ).text

    Args:
        pilmode: Color mode for PIL. Default to ``RGB``.
        mime_type: Return MIME type of the :code:`starlette.response.Response`, only available when used as output descriptor.

    Returns:
        :obj:`Image`: IO Descriptor that either a :code:`PIL.Image.Image` or a :code:`np.ndarray` representing an image.
    """

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

    def openapi_schema(self) -> Schema:
        return Schema(type="string", format="binary")

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
        try:
            return PIL.Image.open(io.BytesIO(bytes_))
        except PIL.UnidentifiedImageError:
            raise BadInput("Failed reading image file uploaded") from None

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
