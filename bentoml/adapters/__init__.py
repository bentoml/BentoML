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
from bentoml.utils.lazy_loader import LazyLoader

DataframeInput = LazyLoader(
    'DataframeInput', globals(), 'bentoml.adapters.dataframe_input.DataframeInput'
)
TfTensorInput = LazyLoader(
    'TfTensorInput', globals(), 'bentoml.adapters.tensorflow_tensor_input.TfTensorInput'
)
JsonInput = LazyLoader('JsonInput', globals(), 'bentoml.adapters.json_input.JsonInput')
LegacyJsonInput = LazyLoader(
    'LegacyJsonInput', globals(), 'bentoml.adapters.legacy_json_input.LegacyJsonInput'
)
ImageInput = LazyLoader(
    'ImageInput', globals(), 'bentoml.adapters.image_input.ImageInput'
)
MultiImageInput = LazyLoader(
    'MultiImageInput', globals(), 'bentoml.adapters.multi_image_input.MultiImageInput'
)
LegacyImageInput = LazyLoader(
    'LegacyImageInput',
    globals(),
    'bentoml.adapters.legacy_image_input.LegacyImageInput',
)
FastaiImageInput = LazyLoader(
    'FastaiImageInput',
    globals(),
    'bentoml.adapters.fastai_image_input.FastaiImageInput',
)
ClipperBytesInput = LazyLoader(
    'ClipperBytesInput', globals(), 'bentoml.adapters.clipper_input.ClipperBytesInput'
)
ClipperDoublesInput = LazyLoader(
    'ClipperDoublesInput',
    globals(),
    'bentoml.adapters.clipper_input.ClipperDoublesInput',
)
ClipperFloatsInput = LazyLoader(
    'ClipperFloatsInput', globals(), 'bentoml.adapters.clipper_input.ClipperFloatsInput'
)
ClipperIntsInput = LazyLoader(
    'ClipperIntsInput', globals(), 'bentoml.adapters.clipper_input.ClipperIntsInput'
)
ClipperStringsInput = LazyLoader(
    'ClipperStringsInput',
    globals(),
    'bentoml.adapters.clipper_input.ClipperStringsInput',
)
DataframeOutput = LazyLoader(
    'DataframeOutput', globals(), 'bentoml.adapters.dataframe_output.DataframeOutput'
)
TfTensorOutput = LazyLoader(
    'TfTensorOutput',
    globals(),
    'bentoml.adapters.tensorflow_tensor_output.TfTensorOutput',
)
BaseOutputAdapter = LazyLoader(
    'BaseOutputAdapter', globals(), 'bentoml.adapters.base_output.BaseOutputAdapter'
)
DefaultOutput = LazyLoader(
    'DefaultOutput', globals(), 'bentoml.adapters.default_output.DefaultOutput'
)
JsonSerializableOutput = LazyLoader(
    'JsonSerializableOutput',
    globals(),
    'bentoml.adapters.json_output.JsonSerializableOutput',
)


def get_batch_model_supported_input_types():
    return {
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
    "ClipperBytesInput",
    "ClipperDoublesInput",
    "ClipperFloatsInput",
    "ClipperIntsInput",
    "ClipperStringsInput",
    'DefaultOutput',
]
