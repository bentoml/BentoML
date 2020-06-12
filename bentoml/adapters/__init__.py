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
from bentoml.adapters.dataframe_input import DataframeInput
from bentoml.adapters.tensorflow_tensor_handler import TensorflowTensorHandler
from bentoml.adapters.json_handler import JsonHandler
from bentoml.adapters.image_handler import ImageHandler
from bentoml.adapters.legacy_image_handler import LegacyImageHandler
from bentoml.adapters.fastai_image_handler import FastaiImageHandler
from bentoml.adapters.clipper_input import (
    ClipperBytesInput,
    ClipperDoublesInput,
    ClipperFloatsInput,
    ClipperIntsInput,
    ClipperStringsInput,
)

from bentoml.adapters.dataframe_output import DataframeOutput
from bentoml.adapters.tf_tensor_output import TfTensorOutput
from bentoml.adapters.base_output import BaseOutputAdapter
from bentoml.adapters.default_output import DefaultOutput
from bentoml.adapters.json_output import JsonserializableOutput

TfTensorInput = TensorflowTensorHandler
JsonInput = JsonHandler
ImageInput = ImageHandler
LegacyImageInput = LegacyImageHandler
FastaiImageInput = FastaiImageHandler

ClipperStringsInput = ClipperStringsInput


BATCH_MODE_SUPPORTED_INPUT_TYPES = {
    name for name, v in locals().items() if getattr(v, 'BATCH_MODE_SUPPORTED', None)
}


__all__ = [
    "BaseInputAdapter",
    "DataframeInput",
    "TfTensorInput",
    "JsonInput",
    "ImageInput",
    "LegacyImageInput",
    "FastaiImageInput",
    "ClipperBytesInput",
    "ClipperDoublesInput",
    "ClipperFloatsInput",
    "ClipperIntsInput",
    "ClipperStringsInput",
    'DefaultOutput',
    'DataframeOutput',
    'BaseOutputAdapter',
    'TfTensorOutput',
    'JsonserializableOutput',
]
