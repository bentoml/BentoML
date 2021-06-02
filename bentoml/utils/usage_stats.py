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


import sys
import platform
import json
import logging
import os
import uuid
from functools import lru_cache

from bentoml.utils.ruamel_yaml import YAML
from bentoml.utils import ProtoMessageToDict
from bentoml.configuration import _is_pip_installed_bentoml
from bentoml import __version__ as BENTOML_VERSION


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _amplitude_url():
    return "https://api.amplitude.com/httpapi"


@lru_cache(maxsize=1)
def _platform():
    return platform.platform()


@lru_cache(maxsize=1)
def _py_version():
    return "{major}.{minor}.{micro}".format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        micro=sys.version_info.micro,
    )


@lru_cache(maxsize=1)
def _session_id():
    return str(uuid.uuid4())  # uuid that marks current python session


@lru_cache(maxsize=1)
def _api_key():
    if _is_pip_installed_bentoml():
        # Use prod amplitude key
        return '1ad6ee0e81b9666761aebd55955bbd3a'
    else:
        # Use dev amplitude key
        return '7f65f2446427226eb86f6adfacbbf47a'


def _send_amplitude_event(event_type, event_properties):
    """Send event to amplitude
    https://developers.amplitude.com/?objc--ios#http-api-v2-request-format
    """
    event = [
        {
            "event_type": event_type,
            "user_id": _session_id(),
            "event_properties": event_properties,
            "ip": "$remote",
        }
    ]
    event_data = {"api_key": _api_key(), "event": json.dumps(event)}

    try:
        import requests

        return requests.post(_amplitude_url(), data=event_data, timeout=1)
    except Exception as err:  # pylint:disable=broad-except
        # silently fail since this error does not concern BentoML end users
        logger.debug(str(err))


def _get_bento_service_event_properties(bento_service, properties=None):
    bento_service_metadata = bento_service.get_bento_service_metadata_pb()

    if properties is None:
        properties = {}

    artifact_types = set()
    input_types = set()
    output_types = set()

    for artifact in bento_service_metadata.artifacts:
        artifact_types.add(artifact.artifact_type)

    for api in bento_service_metadata.apis:
        input_types.add(api.input_type)
        output_types.add(api.output_type)

    if input_types:
        properties["input_types"] = list(input_types)
    if output_types:
        properties["output_types"] = list(output_types)

    if artifact_types:
        properties["artifact_types"] = list(artifact_types)
    else:
        properties["artifact_types"] = ["NO_ARTIFACT"]

    env_dict = ProtoMessageToDict(bento_service_metadata.env)
    if 'conda_env' in env_dict:
        env_dict['conda_env'] = YAML().load(env_dict['conda_env'])
    properties['env'] = env_dict

    return properties


@lru_cache(maxsize=1)
def _do_not_track() -> bool:
    """Returns if tracking is disabled from the environment variable

    Returns True if and only if the environment variable is defined and has value True.
    The function is cached for better performance.
    """
    return os.environ.get("BENTOML_DO_NOT_TRACK", str(False)).lower() == "true"


def track(event_type, event_properties=None):
    if _do_not_track():
        return  # Usage tracking disabled

    if event_properties is None:
        event_properties = {}

    event_properties['py_version'] = _py_version()
    event_properties["bento_version"] = BENTOML_VERSION
    event_properties["platform_info"] = _platform()

    return _send_amplitude_event(event_type, event_properties)


def track_save(bento_service, extra_properties=None):
    properties = _get_bento_service_event_properties(bento_service, extra_properties)
    return track("save", properties)
