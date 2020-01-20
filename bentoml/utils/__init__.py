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

import re
import os
from io import StringIO
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative


from google.protobuf.json_format import MessageToDict
from ruamel.yaml import YAML

from bentoml import __version__ as BENTOML_VERSION, _version as version_mod
from bentoml.proto import status_pb2

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


def is_url(url):
    try:
        return urlparse(url).scheme in _VALID_URLS
    except Exception:  # pylint:disable=broad-except
        return False


def isidentifier(s):
    """
    Return true if string is in a valid python identifier format:

    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    try:
        return s.isidentifier()
    except AttributeError:
        # str#isidentifier is only available in python 3
        return re.match(r"[A-Za-z_][A-Za-z_0-9]*\Z", s) is not None


def dump_to_yaml_str(yaml_dict):
    yaml = YAML()
    string_io = StringIO()
    yaml.dump(yaml_dict, string_io)
    return string_io.getvalue()


def pb_to_yaml(message):
    message_dict = MessageToDict(message)
    return dump_to_yaml_str(message_dict)


def ProtoMessageToDict(protobuf_msg, **kwargs):
    if 'preserving_proto_field_name' not in kwargs:
        kwargs['preserving_proto_field_name'] = True

    return MessageToDict(protobuf_msg, **kwargs)


def _is_pypi_release():
    is_installed_package = hasattr(version_mod, 'version_json')
    is_tagged = not BENTOML_VERSION.startswith('0+untagged')
    is_clean = not version_mod.get_versions()['dirty']
    return is_installed_package and is_tagged and is_clean


# this is for BentoML developer to create BentoService containing custom develop
# branches of BentoML library, it gets triggered when BentoML module is installed
# via "pip install --editable ."
def _is_bentoml_in_develop_mode():
    import importlib

    (module_location,) = importlib.util.find_spec('bentoml').submodule_search_locations

    setup_py_path = os.path.abspath(os.path.join(module_location, '..', 'setup.py'))
    return not _is_pypi_release() and os.path.isfile(setup_py_path)


# This function assume the status is not status.OK
def status_pb_to_error_code_and_message(pb_status):
    assert pb_status.status_code != status_pb2.Status.OK
    error_code = status_pb2.Status.Code.Name(pb_status.status_code)
    error_message = pb_status.error_message
    return error_code, error_message
