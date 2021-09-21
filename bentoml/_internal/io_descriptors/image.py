import io
import typing as t

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

import bentoml._internal.constants as const
from bentoml.exceptions import BentoMLException

from ..utils import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:  # pragma: no cover # pylint: disable=unused-import
    import numpy as np
    import PIL.Image
else:
    np = LazyLoader("np", globals(), "numpy")
    # TODO: use pillow-simd by default instead of pillow for better performance
    _exc = const.IMPORT_ERROR_MSG.format(
        fwr="Pillow",
        module=f"{__name__}.Image",
        inst="`pip install Pillow`",
    )
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=_exc)
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image", exc_msg=_exc)


DEFAULT_PIL_MODE = "RGB"


class Image(IODescriptor):
    def __init__(self, pilmode=DEFAULT_PIL_MODE):
        self._pilmode = pilmode

    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    async def from_http_request(self, request: Request) -> "PIL.Image":
        form = await request.form()
        contents = await form.upload_file.read()
        return PIL.Image.open(io.BytesIO(contents), mode=self._pilmode)

    async def to_http_response(
        self, obj: t.Union["np.ndarray", "PIL.Image"]
    ) -> Response:
        if isinstance(obj, np.ndarray):
            image = PIL.Image.fromarray(obj)
        elif isinstance(PIL.Image):
            image = obj
        else:
            raise BentoMLException(
                f"Unsupported Image type received: {obj}, bentoml.io.Image supports "
                f"only 'np.ndarray' and 'PIL.image'"
            )

        # Support other return types?
        ret = io.BytesIO()
        image.save(ret, "PNG")
        return StreamingResponse(ret, media_type="image/png")
