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

import json
import traceback
from typing import BinaryIO, Iterable, Sequence, Tuple

from bentoml.adapters.multi_file_input import MultiFileInput
from bentoml.adapters.utils import (
    check_file_extension,
    get_default_accept_image_formats,
)
from bentoml.types import InferenceTask, JsonSerializable, Optional
from bentoml.utils.lazy_loader import LazyLoader

# BentoML optional dependencies, using lazy load to avoid ImportError
imageio = LazyLoader('imageio', globals(), 'imageio')
numpy = LazyLoader('numpy', globals(), 'numpy')


ApiFuncArgs = Tuple[Sequence[numpy.ndarray], Sequence[Optional[JsonSerializable]]]
AnnoImgTask = InferenceTask[Tuple[BinaryIO, BinaryIO]]  # image file bytes, json bytes


class AnnotatedImageInput(MultiFileInput):
    """Transform incoming image data from http request, cli or lambda event into a
    numpy array, while allowing an optional JSON file for image annotations (such
    as object bounding boxes, class labels, etc.)

    Transforms input image file into a numpy array, and loads JSON file as
    a JSON serializable Python object, providing them to user-defined
    API functions.

    Args:
        image_input_name (string): An acceptable input name for HTTP request
            and function keyword argument. Default value is "image"
        annotation_input_name (string): An acceptable input name for HTTP request
            and function keyword argument. Default value is "annotations"
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
        ImportError: imageio package is required to use AnnotatedImageInput

    Example:

        >>> from bentoml import BentoService, api, artifacts
        >>> from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
        >>> from bentoml.adapters import AnnotatedImageInput
        >>>
        >>> CLASS_NAMES = ['cat', 'dog']
        >>>
        >>> @artifacts([TensorflowSavedModelArtifact('classifier')])
        >>> class PetClassification(BentoService):
        >>>    @api(input=AnnotatedImageInput(), batch=True)
        >>>    def predict(
        >>>            self,
        >>>            image_list: 'Sequence[numpy.ndarray]',
        >>>            annotations_list: 'Sequence[JsonSerializable]',
        >>>        ):
        >>>        cropped_pets = some_pet_finder(image_list, annotations_list)
        >>>        results = self.artifacts.classifer.predict(cropped_pets)
        >>>        return [CLASS_NAMES[r] for r in results]
        >>>

        The endpoint could then be used with an HTML form that sends multipart data,
        like the example below

        >>> <form action="http://localhost:8000" method="POST"
        >>>       enctype="multipart/form-data">
        >>>     <input name="image" type="file">
        >>>     <input name="annotations" type="file">
        >>>     <input type="submit">
        >>> </form>

        Or the following cURL command

        >>> curl -F image=@image.png
        >>>      -F annotations=@annotations.json
        >>>      http://localhost:8000/predict
    """

    def __init__(
        self,
        image_input_name="image",
        annotation_input_name="annotations",
        pilmode="RGB",
        accept_image_formats=None,
        **base_kwargs,
    ):
        assert imageio, "`imageio` dependency can be imported"
        input_names = [image_input_name, annotation_input_name]
        super().__init__(input_names=input_names, allow_none=True, **base_kwargs)

        self.image_input_name = image_input_name
        self.annotation_input_name = annotation_input_name
        self.pilmode = pilmode
        self.accept_image_formats = (
            accept_image_formats or get_default_accept_image_formats()
        )

    @property
    def pip_dependencies(self):
        return ["imageio"]

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
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        self.image_input_name: {"type": "string", "format": "binary"},
                        self.annotation_input_name: {
                            "type": "string",
                            "format": "binary",
                        },
                    },
                }
            },
        }

    def extract_user_func_args(self, tasks: Iterable[AnnoImgTask]) -> ApiFuncArgs:
        image_arrays = []
        json_objs = []
        for task in tasks:
            try:
                image_file, json_file = task.data
                assert image_file is not None
                assert getattr(image_file, "name", None)
                assert check_file_extension(image_file.name, self.accept_image_formats)
                image_array = imageio.imread(image_file, pilmode=self.pilmode)
                image_arrays.append(image_array)
                if json_file is not None:
                    json_objs.append(json.load(json_file))
                else:
                    json_objs.append(None)
            except AssertionError:
                task.discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} "
                    f"Input image file must be in supported format list: "
                    f"{self.accept_image_formats}",
                )
            except UnicodeDecodeError:
                task.discard(
                    http_status=400, err_msg="JSON must be in unicode",
                )
            except json.JSONDecodeError:
                task.discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} "
                    f"received invalid JSON file",
                )
            except Exception:  # pylint: disable=broad-except
                err = traceback.format_exc()
                task.discard(
                    http_status=500,
                    err_msg=f"BentoML#{self.__class__.__name__} "
                    f"Internal Server Error: {err}",
                )
        return tuple(image_arrays), tuple(json_objs)
