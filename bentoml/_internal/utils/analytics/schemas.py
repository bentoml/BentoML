import os
import re
import typing as t
from abc import ABC
from typing import TYPE_CHECKING
from datetime import datetime
from platform import platform
from platform import python_version
from functools import lru_cache

import yaml
import attrs
import click
import psutil
import attrs.converters

from ...types import Tag
from ..lazy_loader import LazyLoader
from ...configuration import BENTOML_VERSION
from ...configuration.containers import CLIENT_ID_PATH
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
            map(str.lower, re.findall(r"[A-Z][^A-Z]*", self.__class__.__name__)[:-1])
        )


def from_ns_to_ms(duration: t.Union[int, float]) -> float:
    return duration / 1e6


@attrs.define
class CliEvent(EventMeta):

    command_group: str
    command_name: str
    duration_in_ms: float = attrs.field(converter=from_ns_to_ms)
    error_type: t.Optional[str] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("")
    )
    error_message: t.Optional[str] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("")
    )

    return_code: t.Optional[int] = attrs.field(
        default=None, converter=attrs.converters.default_if_none(0)
    )

    @property
    def _track_event_name(self):
        return f"{self.command_name}_{super()._track_event_name}"


@attrs.define
class BentoBuildEvent(EventMeta):
    bento_creation_timestamp: datetime
    bento_size_in_kb: float = attrs.field(converter=convert_to_kb)
    model_size_in_kb: float = attrs.field(converter=convert_to_kb)

    num_of_models: int = attrs.field(
        default=None, converter=attrs.converters.default_if_none(0)
    )
    num_of_runners: int = attrs.field(
        default=None, converter=attrs.converters.default_if_none(0)
    )
    model_types: t.List[str] = attrs.field(factory=list)
    runner_types: t.List[str] = attrs.field(factory=list)


@attrs.define
class CLIContext:
    command_group: str = attrs.field(init=False)
    event: CliEvent = attrs.field(init=False)
    custom_event_mapping: "t.Dict[str, GenericFunction[t.Any]]" = attrs.field(
        init=False
    )


pass_cli_context = click.make_pass_decorator(CLIContext, ensure=True)


def command_tree(cli: click.Group) -> t.Generator[t.Tuple[str, str], None, None]:
    for cmd in cli.commands.values():
        if isinstance(cmd, click.Group):
            yield from command_tree(cmd)
        else:
            yield cmd.name, f"{cli.name}_{cmd.name}"


@lru_cache(maxsize=1)
def get_event_properties_mapping(
    cli: click.Group,
) -> "t.Dict[str, GenericFunction[t.Any]]":
    getter_func = "get_event_{}"
    getter = LazyLoader(
        "getter", globals(), "bentoml._internal.utils.analytics.get_event_tracking"
    )
    return {
        cmd_name: getattr(getter, getter_func.format(full_cli_name))
        for cmd_name, full_cli_name in command_tree(cli)
        if getter_func.format(full_cli_name) in getter.__all__
    }


@attrs.define
class ModelSaveEvent(EventMeta):
    module: str

    model_tag: Tag

    model_creation_timestamp: datetime
    model_size_in_kb: float = attrs.field(converter=convert_to_kb)


@attrs.define
class ServeDevStartEvent(EventMeta):
    serve_id: str
    serve_started_timestamp: datetime


@attrs.define
class ServeDevUpdateEvent(EventMeta):
    serve_id: str
    triggered_at: datetime


@attrs.define
class ServeDevEndEvent(EventMeta):
    serve_id: str
    triggered_at: datetime


@attrs.define
class ServeStartEvent(EventMeta):
    serve_id: str
    serve_started_timestamp: datetime

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
    triggered_at: datetime


@attrs.define
class ServeEndEvent(EventMeta):
    serve_id: str
    triggered_at: datetime


@attrs.define
class TrackingPayload:
    session_id: str
    event_properties: EventMeta
    common_properties: CommonProperties
    event_type: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.event_type = self.event_properties._track_event_name  # type: ignore
