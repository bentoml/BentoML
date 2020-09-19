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

from typing import BinaryIO, Iterable, Sequence, Tuple

from bentoml.adapters.file_input import FileInput
from bentoml.adapters.utils import (
    check_file_extension,
    get_default_accept_image_formats,
)
from bentoml.types import InferenceTask
from bentoml.utils.lazy_loader import LazyLoader

# BentoML optional dependencies, using lazy load to avoid ImportError
imageio = LazyLoader('imageio', globals(), 'imageio')
numpy = LazyLoader('numpy', globals(), 'numpy')


ApiFuncArgs = Tuple[
    Sequence['numpy.ndarray'],
]


class ImageInput(FileInput):
    """Transform incoming image data from http request, cli or lambda event into numpy
    array.

    Handle incoming image data from different sources, transform them into numpy array
    and pass down to user defined API functions

    * If you want to operate raw image file stream or PIL.Image objects, use lowlevel
        alternative FileInput.

    Args:
        accept_image_formats (string[]):  A list of acceptable image formats.
            Default value is loaded from bentoml config
            'apiserver/default_image_input_accept_file_extensions', which is
            set to ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp'] by default.
            List of all supported format can be found here:
            https://imageio.readthedocs.io/en/stable/formats.html
        pilmode (string): The pilmode to be used for reading image file into numpy
            array. Default value is 'RGB'.  Find more information at:
            https://imageio.readthedocs.io/en/stable/format_png-pil.html

    Raises:
        ImportError: imageio package is required to use ImageInput

    Example:

        >>> from bentoml import BentoService, api, artifacts
        >>> from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
        >>> from bentoml.adapters import ImageInput
        >>>
        >>> CLASS_NAMES = ['cat', 'dog']
        >>>
        >>> @artifacts([TensorflowSavedModelArtifact('classifier')])
        >>> class PetClassification(BentoService):
        >>>     @api(input=ImageInput(), batch=True)
        >>>     def predict(self, image_ndarrays):
        >>>         results = self.artifacts.classifer.predict(image_ndarrays)
        >>>         return [CLASS_NAMES[r] for r in results]
    """

    def __init__(
        self, accept_image_formats=None, pilmode="RGB", **base_kwargs,
    ):
        assert imageio, "`imageio` dependency can be imported"

        super().__init__(**base_kwargs)
        if 'input_names' in base_kwargs:
            raise TypeError(
                "ImageInput doesn't take input_names as parameters since bentoml 0.8."
                "Update your Service definition "
                "or use LegacyImageInput instead(not recommended)."
            )

        self.pilmode = pilmode
        self.accept_image_formats = set(
            accept_image_formats or get_default_accept_image_formats()
        )

    @property
    def config(self):
        return {
            # Converting to list, google.protobuf.Struct does not work with tuple type
            "accept_image_formats": list(self.accept_image_formats),
            "pilmode": self.pilmode,
        }

    @property
    def request_schema(self):
        return {
            "image/*": {"schema": {"type": "string", "format": "binary"}},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image_file": {"type": "string", "format": "binary"}
                    },
                }
            },
        }

    @property
    def pip_dependencies(self):
        return ["imageio"]

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[BinaryIO]]
    ) -> ApiFuncArgs:
        img_list = []
        for task in tasks:
            if getattr(task.data, "name", None) and not check_file_extension(
                task.data.name, self.accept_image_formats
            ):
                task.discard(
                    http_status=400,
                    err_msg=f"Current service only accepts "
                    f"{self.accept_image_formats} formats",
                )
                continue
            try:
                img_array = imageio.imread(task.data, pilmode=self.pilmode)
                img_list.append(img_array)
            except ValueError as e:
                task.discard(http_status=400, err_msg=str(e))

        return (img_list,)
