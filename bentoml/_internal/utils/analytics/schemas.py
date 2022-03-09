import os
import re
import typing as t
from abc import ABC
from datetime import datetime
from platform import platform
from platform import python_version
from functools import lru_cache

import yaml
import attrs
import psutil
import attrs.converters

from ...types import Tag
from ...configuration import BENTOML_VERSION
from ...configuration.containers import CLIENT_ID_PATH
from ...yatai_rest_api_client.config import get_config_path
from ...yatai_rest_api_client.config import get_current_context


@lru_cache(maxsize=1)
def get_platform() -> str:
    return platform(aliased=True)


@lru_cache(maxsize=1)
def get_python_version() -> str:
    return python_version()


@lru_cache(maxsize=1)
def get_client_id() -> "t.Optional[ClientInfo]":  # pragma: no cover
    if os.path.exists(CLIENT_ID_PATH):
        with open(CLIENT_ID_PATH, "r", encoding="utf-8") as f:
            client_info = yaml.safe_load(f)
        return ClientInfo(**client_info)


@attrs.define
class ClientInfo:
    client_id: str
    client_creation_timestamp: datetime


@lru_cache(maxsize=1)
def get_yatai_user_email() -> t.Optional[str]:
    if os.path.exists(get_config_path()):
        return get_current_context().email


def convert_to_kb(size: t.Union[int, float]) -> float:
    return size / 1024


@attrs.define
class CommonProperties:

    # hardware-related
    platform: str = attrs.field(init=False)
    python_version: str = attrs.field(init=False)
    num_threads: int = attrs.field(init=False)
    memory_usage_percent: float = attrs.field(init=False)
    total_memory_in_kb: float = attrs.field(init=False)

    # yatai-related
    yatai_user_email: t.Optional[str] = attrs.field(init=False)

    # bentoml-related
    client_id: t.Optional[ClientInfo] = attrs.field(init=False)
    bentoml_version: str = attrs.field(default=BENTOML_VERSION)

    def __attrs_post_init__(self):
        self.client_id = get_client_id()
        self.yatai_user_email = get_yatai_user_email()

        self.platform = get_platform()
        self.python_version = get_python_version()

        proc = psutil.Process(os.getpid())
        with proc.oneshot():
            self.memory_usage_percent = proc.memory_percent()
            self.total_memory_in_kb = convert_to_kb(psutil.virtual_memory().total)  # type: ignore
            self.num_threads = proc.num_threads()


class EventMeta(ABC):
    @property
    def _track_event_name(self):
        return "_".join(
            map(str.lower, re.findall(r"[A-Z][^A-Z]*", self.__class__.__name__))
        )


@attrs.define
class CliEvent(EventMeta):

    command_group: str
    command_name: str
    error_type: str = ""
    error_message: str = ""

    return_code: int = 0
    duration: float = 0

    @property
    def _track_event_name(self):
        return f"{self.command_name}_{super()._track_event_name}"


@attrs.define
class ModelSaveEvent(EventMeta):
    module: str

    model_tag: Tag

    model_creation_timestamp: datetime
    model_size_in_kb: float = attrs.field(converter=convert_to_kb)


@attrs.define
class BentoBuildEvent(EventMeta):
    bento_tag: Tag
    bento_creation_timestamp: datetime
    bento_size_in_kb: float = attrs.field(converter=convert_to_kb)

    model_tags: t.List[Tag] = attrs.field(
        factory=list, converter=lambda x: [str(i) for i in x]
    )
    model_types: t.List[str] = attrs.field(factory=list)


def check_local_or_packaged_in_bento(identifier: str) -> str:
    if re.match(r"^[A-Za-z_][A-Za-z_0-9]*:[A-Za-z0-9.+-_]*$", identifier) is not None:
        return "packaged"
    else:
        return "local"


@attrs.define
class BentoServeDevelopmentOnStartupEvent(EventMeta):
    serve_id: str
    serve_creation_timestamp: datetime
    bento_identifier: str
    serve_location: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.serve_location = check_local_or_packaged_in_bento(self.bento_identifier)


@attrs.define
class BentoServeDevelopmentScheduledEvent(EventMeta):
    serve_id: str
    bento_identifier: str
    triggered_at: datetime


@attrs.define
class BentoServeDevelopmentOnShutdownEvent(EventMeta):
    serve_id: str
    bento_identifier: str
    triggered_at: datetime


@attrs.define
class BentoServeProductionOnStartupEvent(EventMeta):
    serve_id: str
    serve_creation_timestamp: datetime

    bento_identifier: str
    bento_tag: t.Optional[str] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("")
    )
    bento_creation_timestamp: t.Optional[datetime] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("")
    )

    model_tags: t.List[Tag] = attrs.field(
        factory=list, converter=lambda x: [str(i) for i in x]
    )
    model_types: t.List[str] = attrs.field(factory=list)

    serve_location: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.serve_location = check_local_or_packaged_in_bento(self.bento_identifier)


@attrs.define
class BentoServeProductionScheduledEvent(EventMeta):
    serve_id: str
    bento_identifier: str
    triggered_at: datetime


@attrs.define
class BentoServeProductionOnShutdownEvent(EventMeta):
    serve_id: str
    bento_identifier: str
    triggered_at: datetime


@attrs.define
class TrackingPayload:
    session_id: str
    event_properties: EventMeta
    common_properties: CommonProperties
    event_type: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.event_type = self.event_properties._track_event_name  # type: ignore
