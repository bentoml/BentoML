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
from typing import BinaryIO, Iterable, Sequence, Tuple

from bentoml.adapters.multi_file_input import MultiFileInput
from bentoml.adapters.utils import (
    check_file_extension,
    get_default_accept_image_formats,
)
from bentoml.types import InferenceTask
from bentoml.utils.lazy_loader import LazyLoader

# BentoML optional dependencies, using lazy load to avoid ImportError
imageio = LazyLoader('imageio', globals(), 'imageio')
numpy = LazyLoader('numpy', globals(), 'numpy')


MultiImgTask = InferenceTask[Tuple[BinaryIO, ...]]  # image file bytes, json bytes
ApiFuncArgs = Tuple[Sequence['numpy.ndarray'], ...]


class MultiImageInput(MultiFileInput):
    """
    Transform incoming image data from http request, cli or lambda event into imageio
    arrays (a subclass of numpy.ndarray that has a meta attribute) and pass down to
    user defined API functions.

    Parameters
    ----------
    input_names : List[str]
        A tuple of acceptable input name for HTTP request.
        Default value is (image,)

    accept_image_formats : List[str]
        A list of acceptable image formats.
        Default value is loaded from bentoml config
        'apiserver/default_image_input_accept_file_extensions', which is
        set to ['.jpg', '.png', '.jpeg', '.tiff', '.webp', '.bmp'] by default.
        List of all supported format can be found here:
        https://imageio.readthedocs.io/en/stable/formats.html

    pilmode : str
        The pilmode to be used for reading image file into numpy
        array. Default value is 'RGB'.  Find more information at:
        https://imageio.readthedocs.io/en/stable/format_png-pil.html

    Raises
    ----------
    ImportError: imageio package is required to use MultiImageInput

    Examples
    ----------

    Service using MultiImageInput:

    .. code-block:: python

        from bentoml import BentoService
        import bentoml

        class MyService(BentoService):
            @bentoml.api(
                input=MultiImageInput(input_names=('imageX', 'imageY')), batch=True)
            def predict(self, image_groups):
                for image_group in image_groups:
                    image_array_x = image_group['imageX']
                    image_array_y = image_group['imageY']

    Query the endpoint with HTTP request performed by cURL::

        curl -i \\
          -F imageX=@testx.jpg \\
          -F imageY=@testy.jpg \\
          localhost:5000/predict

    OR by an HTML form that sends multipart data:

    .. code-block:: html

        <form action="http://localhost:8000" method="POST"
              enctype="multipart/form-data">
            <input name="imageX" type="file">
            <input name="imageY" type="file">
            <input type="submit">
        </form>

    OR by Python requests:

    .. code-block:: python

        import requests

        with open("testx.jpg", "rb") as f:
            image_bytes1 = f.read()
        with open("testy.jpg", "rb") as f:
            image_bytes2 = f.read()

        files = {
            "imageX": ("testx.jpg", image_bytes1),
            "imageY": ("testy.jpg", image_bytes2),
        }
        response = requests.post(your_url, files=files)

    .. code-block:: python

        import requests
        import PIL

        pil_image_x = PIL.Image.open('testx.jpg')
        pil_image_y = PIL.Image.open('testy.jpg')

        image_bytes1 = pil_image_x.tobytes()  # from PIL.Image
        image_bytes1 = pil_image_y.tobytes()

        files = {
            "imageX": ("testx.jpg", image_bytes1),
            "imageY": ("testy.jpg", image_bytes2),
        }
        response = requests.post(your_url, files=files)

    Query with CLI command::

        bentoml run PyTorchFashionClassifier:latest predict \\
          --input-file-imageX testx.jpg \\
          --input-file-imageY testy.jpg

    OR infer all file pairs under a folder with ten pairs each batch::

        bentoml run PyTorchFashionClassifier:latest predict --max-batch-size 10 \\
          --input-file-imageX folderx/*.jpg \\
          --input-file-imageY foldery/*.jpg

    Note: jpg files and json files should be in same prefix like this::

        folderx:
            - 1.jpg
            - 2.jpg
            ...
        foldery:
            - 1.jpg
            - 2.jpg
            ...


    Or the following cURL command

    >>> curl -F imageX=@image_file_x.png
    >>>      -F imageY=@image_file_y.jpg
    >>>      http://localhost:8000
    """

    def __init__(
        self,
        input_names=("image",),
        pilmode="RGB",
        accept_image_formats=None,
        **base_kwargs,
    ):
        assert imageio, "`imageio` dependency can be imported"

        super().__init__(input_names=input_names, allow_none=True, **base_kwargs)

        self.pilmode = pilmode
        self.accept_image_formats = (
            accept_image_formats or get_default_accept_image_formats()
        )

    @property
    def pip_dependencies(self):
        return ["imageio"]

    @property
    def config(self):
        return dict(
            super().config,
            accept_image_formats=list(self.accept_image_formats),
            pilmode=self.pilmode,
        )

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
                    imageio.imread(f, pilmode=self.pilmode) for f in task.data
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

    def extract_user_func_args(self, tasks: Iterable[MultiImgTask]) -> ApiFuncArgs:
        args = tuple(map(tuple, zip(*self._extract(tasks))))
        if not args:
            args = (tuple(),) * len(self.input_names)
        return args
