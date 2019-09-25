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


import sys
import platform
import json
import logging
import time
import atexit

import uuid
import requests

from bentoml.utils import _is_pypi_release
from bentoml import config
from bentoml import __version__ as BENTOML_VERSION


logger = logging.getLogger(__name__)

AMPLITUDE_URL = "https://api.amplitude.com/httpapi"
PLATFORM = platform.platform()
PY_VERSION = "{major}.{minor}.{micro}".format(
    major=sys.version_info.major,
    minor=sys.version_info.minor,
    micro=sys.version_info.micro,
)
SESSION_ID = str(uuid.uuid4())  # uuid that marks current python session


# Use dev amplitude key
API_KEY = '7f65f2446427226eb86f6adfacbbf47a'
if _is_pypi_release():
    # Use prod amplitude key
    API_KEY = '1ad6ee0e81b9666761aebd55955bbd3a'


def _send_amplitude_event(event_type, event_properties):
    """Send event to amplitude
    https://developers.amplitude.com/?java#keys-for-the-event-argument
    """
    event = [
        {
            "event_type": event_type,
            "user_id": SESSION_ID,
            "event_properties": event_properties,
        }
    ]
    event_data = {"api_key": API_KEY, "event": json.dumps(event)}

    try:
        return requests.post(AMPLITUDE_URL, data=event_data, timeout=1)
    except Exception as err:  # pylint:disable=broad-except
        # silently fail since this error does not concern BentoML end users
        logger.debug(str(err))


def _get_bento_service_event_properties(bento_service, properties=None):
    if properties is None:
        properties = {}

    artifact_types = []
    handler_types = []

    for artifact in bento_service.artifacts.items():
        artifact_instance = artifact[1]
        artifact_types.append(artifact_instance.__class__.__name__)

    for api in bento_service.get_service_apis():
        handler_types.append(api.handler.__class__.__name__)

    properties["handler_types"] = handler_types
    properties["artifact_types"] = artifact_types
    properties["env"] = bento_service.env.to_dict()
    return properties


def track(event_type, event_properties=None):
    if not config().getboolean("core", "usage_tracking"):
        return  # Usage tracking disabled

    if event_properties is None:
        event_properties = {}

    event_properties['py_version'] = PY_VERSION
    event_properties["bento_version"] = BENTOML_VERSION
    event_properties["platform_info"] = PLATFORM

    return _send_amplitude_event(event_type, event_properties)


def track_save(bento_service, extra_properties=None):
    properties = _get_bento_service_event_properties(bento_service, extra_properties)
    return track("save", properties)


def track_load_start():
    return track('load-start', {})


def track_load_finish(bento_service):
    properties = _get_bento_service_event_properties(bento_service)
    return track("load", properties)


def track_cli(command, deploy_platform=None):
    properties = {}
    if deploy_platform is not None:
        properties['platform'] = deploy_platform
    return track('cli-' + command, properties)


def track_server(server_type, extra_properties=None):
    properties = extra_properties or {}

    # track server start event
    track('server-{server_type}-start'.format(server_type=server_type), properties)

    start_time = time.time()

    @atexit.register
    def log_exit():  # pylint: disable=unused-variable
        # track server stop event
        duration = time.time() - start_time
        properties['uptime'] = int(duration)
        return track(
            'server-{server_type}-stop'.format(server_type=server_type), properties
        )
