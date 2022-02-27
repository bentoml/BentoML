import secrets
import os
import json
import logging
import platform
from functools import lru_cache

import attr

from bentoml import __version__ as BENTOML_VERSION

from bentoml._internal.configuration.containers import BentoMLContainer
from simple_di import Provide, inject

logger = logging.getLogger(__name__)


def get_serve_id():
    return secrets.token_urlsafe(32)


@lru_cache(maxsize=1)
def do_not_track() -> bool:
    """Returns if tracking is disabled from the environment variable

    Returns True if and only if the environment variable is defined and has value True.
    The function is cached for better performance.
    """
    return os.environ.get("BENTOML_DO_NOT_TRACK", str(False)).lower() == "true"


@lru_cache(maxsize=1)
def get_platform() -> str:
    return platform.platform(aliased=True)


@lru_cache(maxsize=1)
def get_python_version() -> str:
    return platform.python_version()


@attr.define()
class CommonField:...



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


def track(event_type, event_properties=None):
    if do_not_track():
        return

    if event_properties is None:
        event_properties = {}

    event_properties["py_version"] = get_python_version()
    event_properties["bento_version"] = BENTOML_VERSION
    event_properties["platform_info"] = get_platform()

    return _send_amplitude_event(event_type, event_properties)
