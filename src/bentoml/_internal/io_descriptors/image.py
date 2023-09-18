from __future__ import annotations

import io
import typing as t
import functools
from urllib.parse import quote

from starlette.requests import Request
from multipart.multipart import parse_options_header
from starlette.responses import Response
from starlette.datastructures import UploadFile

from .base import IODescriptor
from ..types import LazyType
from ..utils import LazyLoader
from ..utils import resolve_user_filepath
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import InvalidArgument
from ...exceptions import InternalServerError
from ...exceptions import MissingDependencyException
from ...grpc.utils import import_generated_stubs
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import MediaType

PIL_EXC_MSG = "'Pillow' is required to use the Image IO descriptor. Install with 'pip install bentoml[io-image]'."

if t.TYPE_CHECKING:
    from types import UnionType

    import PIL
    import PIL.Image

    from .. import external_typing as ext
    from .base import OpenAPIResponse
    from ..context import ServiceContext as Context
    from ...grpc.v1 import service_pb2 as pb
    from ...grpc.v1alpha1 import service_pb2 as pb_v1alpha1

    _Mode = t.Literal[
        "1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"
    ]
else:
    # NOTE: pillow-simd only benefits users who want to do preprocessing
    # TODO: add options for users to choose between simd and native mode
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=PIL_EXC_MSG)
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image", exc_msg=PIL_EXC_MSG)

    pb, _ = import_generated_stubs("v1")
    pb_v1alpha1, _ = import_generated_stubs("v1alpha1")

# NOTES: we will keep type in quotation to avoid backward compatibility
#  with numpy < 1.20, since we will use the latest stubs from the main branch of numpy.
#  that enable a new way to type hint an ndarray.
ImageType = t.Union["PIL.Image.Image", "ext.NpNDArray"]

DEFAULT_PIL_MODE = "RGB"


PIL_WRITE_ONLY_FORMATS = {"PALM", "PDF"}
READABLE_MIMES: set[str] = None  # type: ignore (lazy constant)
MIME_EXT_MAPPING: dict[str, str] = None  # type: ignore (lazy constant)


@functools.lru_cache(maxsize=1)
def initialize_pillow():
    global MIME_EXT_MAPPING  # pylint: disable=global-statement
    global READABLE_MIMES  # pylint: disable=global-statement

    try:
        import PIL.Image
    except ImportError:
        raise InternalServerError(PIL_EXC_MSG)

    PIL.Image.init()
    MIME_EXT_MAPPING = {v: k for k, v in PIL.Image.MIME.items()}  # type: ignore (lazy constant)
    READABLE_MIMES = {k for k, v in MIME_EXT_MAPPING.items() if v not in PIL_WRITE_ONLY_FORMATS}  # type: ignore (lazy constant)


class Image(
    IODescriptor[ImageType], descriptor_id="bentoml.io.Image", proto_fields=("file",)
):
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
        mime_type: The MIME type of the file type that this descriptor should return. Only relevant when used as an output descriptor.
        allowed_mime_types: A list of MIME types to restrict input to.

    Returns:
        :obj:`Image`: IO Descriptor that either a :code:`PIL.Image.Image` or a :code:`np.ndarray` representing an image.
    """

    def __init__(
        self,
        pilmode: _Mode | None = DEFAULT_PIL_MODE,
        mime_type: str = "image/jpeg",
        *,
        allowed_mime_types: t.Iterable[str] | None = None,
    ):
        initialize_pillow()

        if pilmode is not None and pilmode not in PIL.Image.MODES:  # pragma: no cover
            raise InvalidArgument(
                f"Invalid Image pilmode '{pilmode}'. Supported PIL modes are {', '.join(PIL.Image.MODES)}."
            ) from None

        self._mime_type = mime_type.lower()
        self._allowed_mimes: set[str] = (
            READABLE_MIMES
            if allowed_mime_types is None
            else {mtype.lower() for mtype in allowed_mime_types}
        )
        self._allow_all_images = allowed_mime_types is None

        if self._mime_type not in MIME_EXT_MAPPING:  # pragma: no cover
            raise InvalidArgument(
                f"Invalid Image mime_type '{mime_type}'; supported mime types are {', '.join(PIL.Image.MIME.values())} "
            )

        for mtype in self._allowed_mimes:
            if mtype not in MIME_EXT_MAPPING:  # pragma: no cover
                raise InvalidArgument(
                    f"Invalid Image MIME in allowed_mime_types: '{mtype}'; supported mime types are {', '.join(PIL.Image.MIME.values())} "
                )

            if mtype not in READABLE_MIMES:
                raise InvalidArgument(
                    f"Pillow does not support reading '{mtype}' files."
                )

        self._pilmode: _Mode | None = pilmode
        self._format: str = MIME_EXT_MAPPING[self._mime_type]

    def _from_sample(self, sample: ImageType | str) -> ImageType:
        """
        Create a :class:`~bentoml._internal.io_descriptors.image.Image` IO Descriptor from given inputs.

        Args:
            sample: Given File-like object, or a path to a file.
            pilmode: Optional color mode for PIL. Default to ``RGB``.
            mime_type: The MIME type of the file type that this descriptor should return.
                       If not specified, then ``from_sample`` will try to infer the MIME type
                       from file extension.
            allowed_mime_types: An optional list of MIME types to restrict input to.

        Returns:
            :class:`~bentoml._internal.io_descriptors.image.Image`: IODescriptor from given users inputs.

        Example:

        .. code-block:: python
           :caption: `service.py`

           from __future__ import annotations

           import bentoml
           from typing import Any
           from bentoml.io import Image
           import numpy as np

           input_spec = Image.from_sample("/path/to/image.jpg")
           @svc.api(input=input_spec, output=Image())
           async def predict(input: t.IO[t.Any]) -> t.IO[t.Any]:
               return await runner.async_run(input)

        Raises:
            :class:`InvalidArgument`: If the given sample is not a valid image type.
            :class:`MissingDependencyException`: If ``filetype`` is not installed.
            :class:`BadInput`: Given sample from file can't be parsed with Pillow.
        """
        try:
            from filetype.match import image_match
        except ModuleNotFoundError:
            raise MissingDependencyException(
                "'filetype' is required to use 'from_sample'. Install it with 'pip install bentoml[io-image]'."
            )

        img_type = image_match(sample)
        if img_type is None:
            raise InvalidArgument(f"{sample} is not a valid image file type.")

        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(sample):
            sample = PIL.Image.fromarray(sample)
        elif isinstance(sample, str):
            p = resolve_user_filepath(sample, ctx=None)
            try:
                with open(p, "rb") as f:
                    sample = PIL.Image.open(f)
            except PIL.UnidentifiedImageError as err:
                raise BadInput(f"Failed to parse sample image file: {err}") from None
        self._mime_type = img_type.mime
        return sample

    def to_spec(self) -> dict[str, t.Any]:
        return {
            "id": self.descriptor_id,
            "args": {
                "pilmode": self._pilmode,
                "mime_type": self._mime_type,
                "allowed_mime_types": list(self._allowed_mimes),
            },
        }

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        if "args" not in spec:
            raise InvalidArgument(f"Missing args key in Image spec: {spec}")

        return cls(**spec["args"])

    def input_type(self) -> UnionType:
        return ImageType

    def openapi_schema(self) -> Schema:
        return Schema(type="string", format="binary")

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self):
        pass

    def openapi_request_body(self) -> dict[str, t.Any]:
        return {
            "content": {
                mtype: MediaType(schema=self.openapi_schema())
                for mtype in self._allowed_mimes
            },
            "required": True,
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def openapi_responses(self) -> OpenAPIResponse:
        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {self._mime_type: MediaType(schema=self.openapi_schema())},
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    async def from_http_request(self, request: Request) -> ImageType:
        content_type, _ = parse_options_header(request.headers["content-type"])
        mime_type = content_type.decode().lower()

        bytes_: bytes | str | None = None

        if mime_type == "multipart/form-data":
            form = await request.form()

            found_mimes: list[str] = []

            for val in form.values():
                val_content_type = val.content_type  # type: ignore (bad starlette types)
                if isinstance(val, UploadFile):
                    found_mimes.append(val_content_type)

                if self._allowed_mimes is None:
                    if (
                        val_content_type in MIME_EXT_MAPPING
                        or val_content_type.startswith("image/")
                    ):
                        bytes_ = await val.read()
                        break
                elif val_content_type in self._allowed_mimes:
                    bytes_ = await val.read()
                    break
            else:
                if len(found_mimes) == 0:
                    raise BadInput("no image file found in multipart form")
                else:
                    if self._allowed_mimes is None:
                        raise BadInput(
                            f"no multipart image file (supported images are: {', '.join(MIME_EXT_MAPPING.keys())}, or 'image/*'), got files with content types {', '.join(found_mimes)}"
                        )
                    else:
                        raise BadInput(
                            f"no multipart image file (allowed mime types are: {', '.join(self._allowed_mimes)}), got files with content types {', '.join(found_mimes)}"
                        )

        elif self._allowed_mimes is None:
            if mime_type in MIME_EXT_MAPPING or mime_type.startswith("image/"):
                bytes_ = await request.body()
        elif mime_type in self._allowed_mimes:
            bytes_ = await request.body()
        else:
            if self._allowed_mimes is None:
                raise BadInput(
                    f"unsupported mime type {mime_type}; supported mime types are: {', '.join(MIME_EXT_MAPPING.keys())}, or 'image/*'"
                )
            else:
                raise BadInput(
                    f"mime type {mime_type} is not allowed, allowed mime types are: {', '.join(self._allowed_mimes)}"
                )

        assert bytes_ is not None

        if isinstance(bytes_, str):
            bytes_ = bytes(bytes_, "UTF-8")

        try:
            return PIL.Image.open(io.BytesIO(bytes_))
        except PIL.UnidentifiedImageError as err:
            raise BadInput(f"Failed to parse uploaded image file: {err}") from None

    async def to_http_response(
        self, obj: ImageType, ctx: Context | None = None
    ) -> Response:
        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(obj):
            image = PIL.Image.fromarray(obj, mode=self._pilmode)
        elif LazyType["PIL.Image.Image"]("PIL.Image.Image").isinstance(obj):
            image = obj
        else:
            raise BadInput(
                f"Unsupported Image type received: '{type(obj)}', the Image IO descriptor only supports 'np.ndarray' and 'PIL.Image'."
            ) from None
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

    async def from_proto(self, field: pb.File | pb_v1alpha1.File | bytes) -> ImageType:
        if isinstance(field, bytes):
            content = field
        elif isinstance(field, pb_v1alpha1.File):
            from .file import filetype_pb_to_mimetype_map

            mapping = filetype_pb_to_mimetype_map()
            if field.kind:
                try:
                    mime_type = mapping[field.kind]
                    if mime_type != self._mime_type:
                        raise BadInput(
                            f"Inferred mime_type from 'kind' is '{mime_type}', while '{self!r}' is expecting '{self._mime_type}'",
                        )
                except KeyError:
                    raise BadInput(
                        f"{field.kind} is not a valid File kind. Accepted file kind: {[names for names,_ in pb_v1alpha1.File.FileType.items()]}",
                    ) from None
            if not field.content:
                raise BadInput("Content is empty!") from None
            return PIL.Image.open(io.BytesIO(field.content))
        else:
            assert isinstance(field, pb.File)
            if field.kind and field.kind != self._mime_type:
                raise BadInput(
                    f"MIME type from 'kind' is '{field.kind}', while '{self!r}' is expecting '{self._mime_type}'",
                )
            content = field.content
            if not content:
                raise BadInput("Content is empty!") from None

        return PIL.Image.open(io.BytesIO(content))

    async def to_proto_v1alpha1(self, obj: ImageType) -> pb_v1alpha1.File:
        from .file import mimetype_to_filetype_pb_map

        try:
            kind = mimetype_to_filetype_pb_map()[self._mime_type]
        except KeyError:
            raise BadInput(
                f"{self._mime_type} doesn't have a corresponding File 'kind'"
            ) from None

        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(obj):
            image = PIL.Image.fromarray(obj, mode=self._pilmode)
        elif LazyType["PIL.Image.Image"]("PIL.Image.Image").isinstance(obj):
            image = obj
        else:
            raise BadInput(
                f"Unsupported Image type received: '{type(obj)}', the Image IO descriptor only supports 'np.ndarray' and 'PIL.Image'.",
            ) from None
        ret = io.BytesIO()
        image.save(ret, format=self._format)

        return pb_v1alpha1.File(kind=kind, content=ret.getvalue())

    async def to_proto(self, obj: ImageType) -> pb.File:
        if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(obj):
            image = PIL.Image.fromarray(obj, mode=self._pilmode)
        elif LazyType["PIL.Image.Image"]("PIL.Image.Image").isinstance(obj):
            image = obj
        else:
            raise BadInput(
                f"Unsupported Image type received: '{type(obj)}', the Image IO descriptor only supports 'np.ndarray' and 'PIL.Image'.",
            ) from None
        ret = io.BytesIO()
        image.save(ret, format=self._format)

        return pb.File(kind=self._mime_type, content=ret.getvalue())
