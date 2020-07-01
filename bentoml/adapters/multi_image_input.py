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

from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.image_input import get_default_accept_image_formats


class MultiImageInput(BaseInputAdapter):
    """
    Warning: this is a not yet implemented, follow issue #806 for updates
    Transform incoming images data from http request, cli or lambda event into numpy
    array.

    Args:
        input_names (string[]]): A tuple of acceptable input name for HTTP request.
            Default value is (image,)
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

    Example usage:
        ```
        class MyService(BentoService):
            @bentoml.api(input=MultiImageInput(input_names=('imageX', 'imageY')))
            def predict(self, image_groups):
                for image_group in image_groups:
                    image_array_x = image_group['imageX']
                    image_array_y = image_group['imageY']
        ```
    """

    def __init__(
        self,
        input_names=("image",),
        accept_image_formats=None,
        pilmode="RGB",
        is_batch_input=False,
        **base_kwargs,
    ):
        if is_batch_input:
            raise ValueError('ImageInput can not accpept batch inputs')
        super(MultiImageInput, self).__init__(
            is_batch_input=is_batch_input, **base_kwargs
        )
        self.input_names = input_names
        self.pilmode = pilmode
        self.accept_image_formats = (
            accept_image_formats or get_default_accept_image_formats()
        )

    def handle_batch_request(self, requests, func):
        """Handles an HTTP request, convert it into corresponding data
        format that user API function is expecting, and return API
        function result as the HTTP response to client

        :param requests: List of flask request object
        :param func: user API function
        """
        raise NotImplementedError

    def handle_cli(self, args, func):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        :param func: user API function
        """
        raise NotImplementedError

    def handle_aws_lambda_event(self, event, func):
        """Handles a Lambda event, convert event dict into corresponding
        data format that user API function is expecting, and use API
        function result as response

        :param event: AWS lambda event data of the python `dict` type
        :param func: user API function
        """
        raise NotImplementedError
