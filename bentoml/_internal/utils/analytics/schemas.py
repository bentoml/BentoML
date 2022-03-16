import os
import re
import hmac
import uuid
import typing as t
import hashlib
from abc import ABC
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone
from platform import platform
from platform import python_version
from functools import lru_cache

import yaml
import attrs
import cattr
import psutil
import attrs.converters
from attr import asdict
from simple_di import inject

from ...configuration import BENTOML_VERSION
from ...configuration.containers import BentoMLContainer
from ...yatai_rest_api_client.config import get_config_path
from ...yatai_rest_api_client.config import get_current_context

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    GenericFunction = t.Callable[P, t.Any]


@lru_cache(maxsize=1)
def get_platform() -> str:
    return platform(aliased=True)


@lru_cache(maxsize=1)
def get_python_version() -> str:
    return python_version()


@attrs.define
class ClientInfo:
    client_id: str
    client_creation_timestamp: str


@inject
@lru_cache(maxsize=1)
def get_client_id(
    bentoml_home: str = BentoMLContainer.bentoml_home,
) -> t.Optional[ClientInfo]:
    CLIENT_ID_PATH = os.path.join(bentoml_home, "client_id")

    if os.path.exists(CLIENT_ID_PATH):
        with open(CLIENT_ID_PATH, "r", encoding="utf-8") as f:
            client_info = yaml.safe_load(f)
        return ClientInfo(**client_info)
    else:

        def create_client_id() -> ClientInfo:
            # returns an unique client_id and timestamp in ISO format
            uniq = uuid.uuid1().bytes
            client_id = hmac.new(uniq, digestmod=hashlib.blake2s).hexdigest()
            created_time = datetime.now(timezone.utc).isoformat()

            return ClientInfo(
                client_id=client_id, client_creation_timestamp=created_time
            )

        def create_client_id_file(client_id: ClientInfo) -> None:
            with open(CLIENT_ID_PATH, "w", encoding="utf-8") as f:
                yaml.dump(asdict(client_id), stream=f)

        new_client_id = create_client_id()
        create_client_id_file(new_client_id)
        return new_client_id


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
            map(str.lower, re.findall(r"[A-Z][^A-Z]*", self.__class__.__name__)[:-1])
        )


def from_ns_to_ms(duration: t.Union[int, float]) -> float:
    return duration / 1e6


@attrs.define
class CliEvent(EventMeta):
    cmd_group: str
    cmd_name: str
    duration_in_ms: float = attrs.field(default=0, converter=from_ns_to_ms)
    error_type: t.Optional[str] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("")
    )
    return_code: t.Optional[int] = attrs.field(
        default=None, converter=attrs.converters.default_if_none(0)
    )


@attrs.define
class BentoBuildEvent(CliEvent):
    bento_creation_timestamp: t.Optional[datetime] = attrs.field(default=None)
    bento_size_in_kb: float = attrs.field(default=0, converter=convert_to_kb)
    model_size_in_kb: float = attrs.field(default=0, converter=convert_to_kb)

    num_of_models: int = attrs.field(default=0)
    num_of_runners: int = attrs.field(default=0)
    model_types: t.List[str] = attrs.field(factory=list)
    runner_types: t.List[str] = attrs.field(factory=list)


@attrs.define
class ModelSaveEvent(EventMeta):
    module: str
    model_creation_timestamp: datetime
    model_size_in_kb: float = attrs.field(converter=convert_to_kb)


@attrs.define
class ServeInitEvent(EventMeta):
    serve_id: str
    serve_started_timestamp: datetime
    production: bool
    serve_from_bento: bool = attrs.field(default=True)

    bento_creation_timestamp: t.Optional[datetime] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("")
    )
    num_of_models: int = attrs.field(
        default=None, converter=attrs.converters.default_if_none(0)
    )
    num_of_runners: int = attrs.field(
        default=None, converter=attrs.converters.default_if_none(0)
    )
    model_types: t.List[str] = attrs.field(factory=list)
    runner_types: t.List[str] = attrs.field(factory=list)


@attrs.define
class ServeUpdateEvent(EventMeta):
    serve_id: str
    production: bool
    triggered_at: datetime
    duration_in_seconds: int
    metrics: t.List[str] = attrs.field(factory=list)


ALL_EVENT_TYPES = t.Union[
    CliEvent,
    ModelSaveEvent,
    BentoBuildEvent,
    ServeInitEvent,
    ServeUpdateEvent,
]


def datetime_encoder(time_obj: t.Union[None, str, datetime]) -> t.Optional[str]:
    if not time_obj:
        return None
    if isinstance(time_obj, str):
        return time_obj
    return time_obj.isoformat()


cattr.register_unstructure_hook(datetime, datetime_encoder)


@attrs.define
class TrackingPayload:
    session_id: str
    event_properties: ALL_EVENT_TYPES
    common_properties: CommonProperties
    event_type: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.event_type = self.event_properties._track_event_name  # type: ignore

    def to_dict(self):
        return cattr.unstructure(self)
