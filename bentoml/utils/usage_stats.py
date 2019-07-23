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

import uuid
import requests

from bentoml import config, load
import bentoml

GA_URL = "https://www.google-analytics.com/collect"
AMPLITUDE_URL = "https://api.amplitude.com/httpapi"


def get_artifact_handler_info(bento_service):
    artifact_types = []
    handler_types = []

    for artifact in bento_service.artifacts.items():
        artifact_instance = artifact[1]
        artifact_types.append(artifact_instance.__class__.__name__)

    for api in bento_service.get_service_apis():
        handler_types.append(api.handler.__class__.__name__)

    return {"handler_types": handler_types, "artifact_types": artifact_types}


def track(action, info):
    info["py_version"] = "{major}.{minor}.{micro}".format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        micro=sys.version_info.micro,
    )

    # for operation, the bentoml version is from bentoml.yml
    info["bento_version"] = bentoml.__version__
    info["platform_info"] = platform.platform()

    return send_amplitude_event(action, info)


def track_archive(bento_service):
    # artifacts type, handler type
    if config.getboolean("enable_tracking"):
        info = get_artifact_handler_info(bento_service)
        return track("archive", info)


def track_cli(command, info):
    # command used
    if config.getboolean("enable_tracking"):
        if not isinstance(info, dict):
            info = get_artifact_handler_info(info)

        return track('cli-' + command, info)


def track_deployment(platform_name, archive_path):
    # artifact type, handler type, deployment platform
    if config.getboolean("enable_tracking"):
        try:
            service = load(archive_path)
            info = get_artifact_handler_info(service)
            info['deployment_platform'] = platform_name
            return track("deploy", info)
        except Exception:  # pylint:disable=broad-except
            return


def send_amplitude_event(event, info):
    """Send event to amplitude
    https://developers.amplitude.com/?java#keys-for-the-event-argument
    """

    event_data = {
        "api_key": config['usage_tracking'].get('api_key'),
        "event": [{
            "event_type": event,
            "user_id": str(uuid.uuid4()),
            "event_properties": info
        }],
    }

    try:
        r = requests.post(GA_URL, data=event_data)
        # r = requests.post(GA_URL, params=event_data)
        print(r)
        return
    except Exception:  # pylint:disable=broad-except
        # silently fail
        return
