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

from bentoml.adapters.annotated_image_input import AnnotatedImageInput
from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.base_output import BaseOutputAdapter
from bentoml.adapters.clipper_input import (
    ClipperBytesInput,
    ClipperDoublesInput,
    ClipperFloatsInput,
    ClipperIntsInput,
    ClipperStringsInput,
)
from bentoml.adapters.dataframe_input import DataframeInput
from bentoml.adapters.dataframe_output import DataframeOutput
from bentoml.adapters.default_output import DefaultOutput
from bentoml.adapters.file_input import FileInput
from bentoml.adapters.image_input import ImageInput
from bentoml.adapters.json_input import JsonInput
from bentoml.adapters.json_output import JsonOutput
from bentoml.adapters.multi_file_input import MultiFileInput
from bentoml.adapters.multi_image_input import MultiImageInput
from bentoml.adapters.string_input import StringInput
from bentoml.adapters.tensorflow_tensor_input import TfTensorInput
from bentoml.adapters.tensorflow_tensor_output import TfTensorOutput

__all__ = [
    "BaseInputAdapter",
    'BaseOutputAdapter',
    "DataframeInput",
    'DataframeOutput',
    "TfTensorInput",
    'TfTensorOutput',
    "JsonInput",
    "StringInput",
    'JsonOutput',
    "ImageInput",
    "MultiImageInput",
    "FileInput",
    "MultiFileInput",
    "AnnotatedImageInput",
    "ClipperBytesInput",
    "ClipperDoublesInput",
    "ClipperFloatsInput",
    "ClipperIntsInput",
    "ClipperStringsInput",
    'DefaultOutput',
]
