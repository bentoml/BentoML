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
from bentoml.adapters.tensorflow_tensor_input import TfTensorInput
from bentoml.adapters.json_input import JsonInput
from bentoml.adapters.legacy_json_input import LegacyJsonInput
from bentoml.adapters.image_input import ImageInput
from bentoml.adapters.multi_image_input import MultiImageInput
from bentoml.adapters.legacy_image_input import LegacyImageInput
from bentoml.adapters.fastai_image_input import FastaiImageInput
from bentoml.adapters.file_input import FileInput
from bentoml.adapters.clipper_input import (
    ClipperBytesInput,
    ClipperDoublesInput,
    ClipperFloatsInput,
    ClipperIntsInput,
    ClipperStringsInput,
)

from bentoml.adapters.dataframe_output import DataframeOutput
from bentoml.adapters.tensorflow_tensor_output import TfTensorOutput
from bentoml.adapters.base_output import BaseOutputAdapter
from bentoml.adapters.default_output import DefaultOutput
from bentoml.adapters.json_output import JsonSerializableOutput


BATCH_MODE_SUPPORTED_INPUT_TYPES = {
    name for name, v in locals().items() if getattr(v, 'BATCH_MODE_SUPPORTED', None)
}


__all__ = [
    "BaseInputAdapter",
    'BaseOutputAdapter',
    "DataframeInput",
    'DataframeOutput',
    "TfTensorInput",
    'TfTensorOutput',
    "JsonInput",
    "LegacyJsonInput",
    'JsonSerializableOutput',
    "ImageInput",
    "MultiImageInput",
    "LegacyImageInput",
    "FastaiImageInput",
    "FileInput",
    "ClipperBytesInput",
    "ClipperDoublesInput",
    "ClipperFloatsInput",
    "ClipperIntsInput",
    "ClipperStringsInput",
    'DefaultOutput',
]
