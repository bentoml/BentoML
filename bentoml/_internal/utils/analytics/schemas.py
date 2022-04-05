import os
import re
import uuid
import typing as t
from abc import ABC
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone
from platform import platform
from platform import python_version
from functools import lru_cache

import yaml
import attrs
import psutil
import attrs.converters
from simple_di import inject
from simple_di import Provide

from ...utils import bentoml_cattr
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
    id: str
    creation_timestamp: datetime


@inject
@lru_cache(maxsize=1)
def get_client_info(
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
) -> t.Optional[ClientInfo]:
    CLIENT_INFO_FILE_PATH = os.path.join(bentoml_home, "client_id")

    if os.path.exists(CLIENT_INFO_FILE_PATH):
        with open(CLIENT_INFO_FILE_PATH, "r", encoding="utf-8") as f:
            client_info = yaml.safe_load(f)
        return bentoml_cattr.structure(client_info, ClientInfo)
    else:
        # Create new client id
        new_client_info = ClientInfo(
            id=str(uuid.uuid4()),
            creation_timestamp=datetime.now(timezone.utc),
        )
        # write client info to ~/bentoml/client_id
        with open(CLIENT_INFO_FILE_PATH, "w", encoding="utf-8") as f:
            yaml.dump(attrs.asdict(new_client_info), stream=f)

        return new_client_info


@lru_cache(maxsize=1)
def get_yatai_user_email() -> t.Optional[str]:
    if os.path.exists(get_config_path()):
        return get_current_context().email


@lru_cache(maxsize=1)
def is_interactive() -> bool:
    import __main__ as main

    return not hasattr(main, "__file__")


@lru_cache(maxsize=1)
def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


@attrs.define
class CommonProperties:
    # when the event is triggered
    timestamp: datetime = attrs.field(factory=lambda: datetime.now(timezone.utc))

    # environment related
    platform: str = attrs.field(factory=get_platform)
    bentoml_version: str = attrs.field(default=BENTOML_VERSION)
    python_version: str = attrs.field(factory=get_python_version)
    is_interactive: bool = attrs.field(factory=is_interactive)
    in_notebook: bool = attrs.field(factory=in_notebook)

    # resource related
    memory_usage_percent: float = attrs.field(init=False)
    total_memory_in_mb: float = attrs.field(init=False)

    # client related
    client: ClientInfo = attrs.field(factory=get_client_info)
    yatai_user_email: t.Optional[str] = attrs.field(factory=get_yatai_user_email)

    def __attrs_post_init__(self):
        self.total_memory_in_mb = int(psutil.virtual_memory().total / 1024.0 / 1024.0)
        proc = psutil.Process(os.getpid())
        with proc.oneshot():
            self.memory_usage_percent = proc.memory_percent()


class EventMeta(ABC):
    @property
    def event_name(self):
        # camel case to snake case
        event_name = re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()
        # remove "_event" suffix
        suffix_to_remove = "_event"
        if event_name.endswith(suffix_to_remove):
            event_name = event_name[: -len(suffix_to_remove)]
        return event_name


@attrs.define
class CliEvent(EventMeta):
    cmd_group: str
    cmd_name: str
    duration_in_ms: float = attrs.field(default=0)
    error_type: t.Optional[str] = attrs.field(default=None)
    return_code: t.Optional[int] = attrs.field(default=None)


@attrs.define
class BentoBuildEvent(CliEvent):
    bento_creation_timestamp: t.Optional[datetime] = attrs.field(default=None)
    bento_size_in_kb: float = attrs.field(default=0)
    model_size_in_kb: float = attrs.field(default=0)

    num_of_models: int = attrs.field(default=0)
    num_of_runners: int = attrs.field(default=0)
    model_types: t.List[str] = attrs.field(factory=list)
    runner_types: t.List[str] = attrs.field(factory=list)


@attrs.define
class ModelSaveEvent(EventMeta):
    module: str
    model_size_in_kb: float


@attrs.define
class ServeInitEvent(EventMeta):
    serve_id: str
    production: bool
    serve_from_bento: bool

    bento_creation_timestamp: t.Optional[datetime]
    num_of_models: int = attrs.field(default=0)
    num_of_runners: int = attrs.field(default=0)
    num_of_apis: int = attrs.field(default=0)
    model_types: t.List[str] = attrs.field(factory=list)
    runner_types: t.List[str] = attrs.field(factory=list)
    api_input_types: t.List[str] = attrs.field(factory=list)
    api_output_types: t.List[str] = attrs.field(factory=list)


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


@attrs.define
class TrackingPayload:
    session_id: str
    event_properties: ALL_EVENT_TYPES
    common_properties: CommonProperties
    event_type: str

    def to_dict(self):
        return bentoml_cattr.unstructure(self)
