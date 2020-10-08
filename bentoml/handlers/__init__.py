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

import functools
import logging

from bentoml.adapters import (
    BaseInputAdapter,
    ClipperBytesInput,
    ClipperDoublesInput,
    ClipperFloatsInput,
    ClipperIntsInput,
    ClipperStringsInput,
    DataframeInput,
    JsonInput,
    MultiImageInput,
    TfTensorInput,
)

logger = logging.getLogger(__name__)

logger.warning(
    'bentoml.handlers.* will be deprecated after BentoML 1.0, '
    'use bentoml.adapters.* instead'
)


def deprecated(cls, cls_name):
    class wrapped_cls(cls):
        def __init__(self, *args, **kwargs):
            super(wrapped_cls, self).__init__(*args, **kwargs)
            logger.warning(
                f'{cls_name} will be deprecated after BentoML 1.0, '
                f'use {cls.__name__} instead'
            )

    for attr_name in functools.WRAPPER_ASSIGNMENTS:
        setattr(wrapped_cls, attr_name, getattr(cls, attr_name, None))
    return wrapped_cls


BentoHandler = deprecated(BaseInputAdapter, 'BentoHandler')
DataframeHandler = deprecated(DataframeInput, 'DataframeHandler')
TensorflowTensorHandler = deprecated(TfTensorInput, 'TensorflowTensorHandler')
JsonHandler = deprecated(JsonInput, 'JsonHandler')
ImageHandler = deprecated(MultiImageInput, 'ImageHandler')
ClipperIntsHandler = deprecated(ClipperIntsInput, 'ClipperIntsHandler')
ClipperBytesHandler = deprecated(ClipperBytesInput, 'ClipperBytesHandler')
ClipperDoublesHandler = deprecated(ClipperDoublesInput, 'ClipperDoublesHandler')
ClipperFloatsHandler = deprecated(ClipperFloatsInput, 'ClipperFloatsHandler')
ClipperStringsHandler = deprecated(ClipperStringsInput, 'ClipperStringsHandler')

__all__ = [
    "BentoHandler",
    "DataframeHandler",
    "TensorflowTensorHandler",
    "JsonHandler",
    "ImageHandler",
    "ClipperBytesHandler",
    "ClipperDoublesHandler",
    "ClipperFloatsHandler",
    "ClipperIntsHandler",
    "ClipperStringsHandler",
]
