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

import numpy as np

# Clipper Util Variables/Functions copied from clipper/containers/python module
INPUT_TYPE_BYTES = 0
INPUT_TYPE_INTS = 1
INPUT_TYPE_FLOATS = 2
INPUT_TYPE_DOUBLES = 3
INPUT_TYPE_STRINGS = 4


def string_to_input_type(input_str):
    input_str = input_str.strip().lower()
    byte_strs = ["b", "bytes", "byte"]
    int_strs = ["i", "ints", "int", "integer", "integers"]
    float_strs = ["f", "floats", "float"]
    double_strs = ["d", "doubles", "double"]
    string_strs = ["s", "strings", "string", "strs", "str"]

    if any(input_str == s for s in byte_strs):
        return INPUT_TYPE_BYTES
    elif any(input_str == s for s in int_strs):
        return INPUT_TYPE_INTS
    elif any(input_str == s for s in float_strs):
        return INPUT_TYPE_FLOATS
    elif any(input_str == s for s in double_strs):
        return INPUT_TYPE_DOUBLES
    elif any(input_str == s for s in string_strs):
        return INPUT_TYPE_STRINGS
    else:
        return -1


def input_type_to_dtype(input_type):
    if input_type == INPUT_TYPE_BYTES:
        return np.dtype(np.int8)
    elif input_type == INPUT_TYPE_INTS:
        return np.dtype(np.int32)
    elif input_type == INPUT_TYPE_FLOATS:
        return np.dtype(np.float32)
    elif input_type == INPUT_TYPE_DOUBLES:
        return np.dtype(np.float64)
    elif input_type == INPUT_TYPE_STRINGS:
        return str


def input_type_to_string(input_type):
    if input_type == INPUT_TYPE_BYTES:
        return "bytes"
    elif input_type == INPUT_TYPE_INTS:
        return "ints"
    elif input_type == INPUT_TYPE_FLOATS:
        return "floats"
    elif input_type == INPUT_TYPE_DOUBLES:
        return "doubles"
    elif input_type == INPUT_TYPE_STRINGS:
        return "string"
