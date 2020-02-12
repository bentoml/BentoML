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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bentoml.handlers.base_handlers import BentoHandler
from bentoml.handlers.dataframe_handler import DataframeHandler
from bentoml.handlers.tensorflow_tensor_handler import TensorflowTensorHandler
from bentoml.handlers.json_handler import JsonHandler
from bentoml.handlers.image_handler import ImageHandler
from bentoml.handlers.fastai_image_handler import FastaiImageHandler
from bentoml.handlers.clipper_handler import (
    ClipperBytesHandler,
    ClipperDoublesHandler,
    ClipperFloatsHandler,
    ClipperIntsHandler,
    ClipperStringsHandler,
)


HANDLER_TYPES_BATCH_MODE_SUPPORTED = {
    name
    for name, v in locals().items()
    if name.endswith("Handler") and getattr(v, 'BATCH_MODE_SUPPORTED', None)
}

__all__ = [
    "BentoHandler",
    "DataframeHandler",
    "TensorflowTensorHandler",
    "JsonHandler",
    "ImageHandler",
    "FastaiImageHandler",
    "ClipperBytesHandler",
    "ClipperDoublesHandler",
    "ClipperFloatsHandler",
    "ClipperIntsHandler",
    "ClipperStringsHandler",
]
