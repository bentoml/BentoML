import gzip
import json
import os

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.types import HTTPRequest, InferenceTask

TF_B64_KEY = "b64"


class B64JsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):  # pylint: disable=method-hidden
        import base64

        if isinstance(o, bytes):
            try:
                return o.decode('utf-8')
            except UnicodeDecodeError:
                return {TF_B64_KEY: base64.b64encode(o).decode("utf-8")}

        try:
            return super(B64JsonEncoder, self).default(o)
        except (TypeError, OverflowError):
            return {"unknown_obj": str(o)}


class NumpyJsonEncoder(B64JsonEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):  # pylint: disable=method-hidden
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()

        if isinstance(o, np.ndarray):
            return o.tolist()

        return super(NumpyJsonEncoder, self).default(o)


class TfTensorJsonEncoder(NumpyJsonEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):  # pylint: disable=method-hidden

        import tensorflow as tf

        # Tensor -> ndarray or object
        if isinstance(o, tf.Tensor):
            if tf.__version__.startswith("1."):
                with tf.compat.v1.Session():
                    return o.numpy()
            else:
                return o.numpy()
        return super(TfTensorJsonEncoder, self).default(o)


def concat_list(lst, batch_flags=None):
    """
    >>> lst = [
        [1],
        [1, 2],
        [1, 2, 3],
        None,
        ]
    >>> concat_list(lst)
    [1, 1, 2, 1, 2, 3], [slice(0, 1), slice(1, 3), slice(3, 6), None]
    """
    slices = [slice(0)] * len(lst)
    datas = []
    row_flag = 0
    for i, r in enumerate(lst):
        if r is None:
            slices[i] = None
            continue
        j = -1
        if batch_flags is None or batch_flags[i]:
            for j, d in enumerate(r):
                datas.append(d)
            slices[i] = slice(row_flag, row_flag + j + 1)
        else:
            datas.append(r)
            slices[i] = row_flag
            j += 1
        row_flag += j + 1
    return datas, slices


def check_file_extension(file_name, accept_ext_list):
    """
    Return False if file's extension is not in the accept_ext_list
    """
    if not file_name:
        return False
    _, extension = os.path.splitext(file_name)
    return extension.lower() in (accept_ext_list or [])


@inject
def get_default_accept_image_formats(
    default_extensions: list = Provide[
        BentoMLContainer.config.adapters.image_input.default_extensions
    ],
):
    """With default bentoML config, this returns:
        ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp']
    """
    return default_extensions


def decompress_gzip_request(method):
    def _method(self, req: HTTPRequest) -> InferenceTask:
        if req.headers.content_encoding in {"gzip", "x-gzip"}:
            # https://tools.ietf.org/html/rfc7230#section-4.2.3
            try:
                req.body = gzip.decompress(req.body)

            except OSError:
                return InferenceTask().discard(
                    http_status=400, err_msg="Gzip decompression error"
                )
            req.headers.pop("content-encoding")
            return method(self, req)
        return method(self, req)

    return _method
