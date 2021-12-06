import os
import sys
import json
import uuid
import logging
import platform
from functools import lru_cache

from bentoml import __version__ as BENTOML_VERSION

from ..configuration import is_pypi_installed_bentoml

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
    if is_pypi_installed_bentoml():
        # Use prod amplitude key
        return "1ad6ee0e81b9666761aebd55955bbd3a"
    else:
        # Use dev amplitude key
        return "7f65f2446427226eb86f6adfacbbf47a"


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

    event_properties["py_version"] = _py_version()
    event_properties["bento_version"] = BENTOML_VERSION
    event_properties["platform_info"] = _platform()

    return _send_amplitude_event(event_type, event_properties)
