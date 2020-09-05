# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
from typing import BinaryIO, Sequence, Tuple

from bentoml.adapters.legacy_image_input import LegacyImageInput
from bentoml.adapters.utils import (
    check_file_extension,
    get_default_accept_image_formats,
)
from bentoml.types import InferenceTask
from bentoml.utils.lazy_loader import LazyLoader

# BentoML optional dependencies, using lazy load to avoid ImportError
fastai = LazyLoader('fastai', globals(), 'fastai')
imageio = LazyLoader('imageio', globals(), 'imageio')
numpy = LazyLoader('numpy', globals(), 'numpy')

MultiImgTask = InferenceTask[Tuple[BinaryIO, ...]]  # image file bytes, json bytes
ApiFuncArgs = Tuple[Sequence['numpy.ndarray'], ...]


class FastaiImageInput(LegacyImageInput):
    """InputAdapter specified for handling image input following fastai conventions
    by passing type fastai.vision.Image to user API function and providing options
    such as div, cls, and after_open

    Args:
        input_names ([str]]): A tuple of acceptable input name for HTTP request.
            Default value is (image,)
        accept_image_formats ([str]):  A list of acceptable image formats.
            Default value is loaded from bentoml config
            'apiserver/default_image_input_accept_file_extensions', which is
            set to ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp'] by default.
            List of all supported format can be found here:
            https://imageio.readthedocs.io/en/stable/formats.html
        convert_mode (str): The pilmode to be used for reading image file into
            numpy array. Default value is 'RGB'.  Find more information at
            https://imageio.readthedocs.io/en/stable/format_png-pil.html
        div (bool): If True, pixel values are divided by 255 to become floats
            between 0. and 1.
        cls (Class): Parameter from fastai.vision ``open_image``, default is
            ``fastai.vision.Image``
        after_open (func): Parameter from fastai.vision ``open_image``, default
            is None

    Raises:
        ImportError: imageio package is required to use FastaiImageInput
        ImportError: fastai package is required to use FastaiImageInput
    """

    BATCH_MODE_SUPPORTED = False

    HTTP_METHODS = ["POST"]

    def __init__(
        self,
        input_names=("image",),
        accept_image_formats=None,
        convert_mode="RGB",
        div=True,
        cls=None,
        after_open=None,
        **base_kwargs,
    ):
        super().__init__(
            input_names=input_names,
            accept_image_formats=accept_image_formats,
            **base_kwargs,
        )

        self.input_names = input_names
        self.convert_mode = convert_mode
        self.div = div
        self.cls = cls
        self.accept_image_formats = (
            accept_image_formats or get_default_accept_image_formats()
        )
        self.after_open = after_open

    @property
    def config(self):
        return {
            # Converting to list, google.protobuf.Struct does not work with tuple type
            "input_names": list(self.input_names),
            "accept_image_formats": list(self.accept_image_formats),
            "convert_mode": self.convert_mode,
            "div": self.div,
            "cls": self.cls.__name__ if self.cls else None,
            "after_open": self.after_open.__name__ if self.after_open else None,
        }

    @property
    def request_schema(self):
        return {
            "image/*": {"schema": {"type": "string", "format": "binary"}},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        filename: {"type": "string", "format": "binary"}
                        for filename in self.input_names
                    },
                }
            },
        }

    @property
    def pip_dependencies(self):
        return ['imageio', 'fastai', 'pandas']

    def _extract(self, tasks):
        for task in tasks:
            if not all(f is not None for f in task.data):
                task.discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} Empty request",
                )
                continue
            try:
                assert all(
                    not getattr(f, "name", None)
                    or check_file_extension(f.name, self.accept_image_formats)
                    for f in task.data
                )
                image_array_tuple = tuple(
                    fastai.vision.open_image(
                        fn=f,
                        convert_mode=self.convert_mode,
                        div=self.div,
                        after_open=self.after_open,
                        cls=self.cls or fastai.vision.Image,
                    )
                    for f in task.data
                )
                yield image_array_tuple
            except ValueError:
                task.discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} "
                    f"Input image decode failed, it must be in supported format list: "
                    f"{self.accept_image_formats}",
                )
            except AssertionError:
                task.discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} "
                    f"Input image file must be in supported format list: "
                    f"{self.accept_image_formats}",
                )
            except Exception:  # pylint: disable=broad-except
                err = traceback.format_exc()
                task.discard(
                    http_status=500,
                    err_msg=f"BentoML#{self.__class__.__name__} "
                    f"Internal Server Error: {err}",
                )

    def extract_user_func_args(self, tasks: Sequence[MultiImgTask]) -> ApiFuncArgs:
        args = tuple(map(tuple, zip(*self._extract(tasks))))
        if not args:
            args = (tuple(),) * len(self.input_names)
        return args
