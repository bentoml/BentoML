from __future__ import annotations

import os
import re
import typing as t
import uuid
from abc import ABC
from datetime import datetime
from datetime import timezone
from functools import lru_cache
from platform import platform
from platform import python_version

import attr
import attr.converters
import psutil
import yaml
from simple_di import Provide
from simple_di import inject

from ...cloud.config import CloudClientConfig
from ...configuration import BENTOML_VERSION
from ...configuration.containers import BentoMLContainer
from ...utils import is_jupyter
from ...utils.cattr import bentoml_cattr

if t.TYPE_CHECKING:
    from pathlib import Path

    P = t.ParamSpec("P")
    GenericFunction = t.Callable[P, t.Any]

# Refer to bentoml/yatai-deployment-operator/common/consts/consts.go
ENV_YATAI_VERSION = "YATAI_T_VERSION"
ENV_YATAI_ORG_UID = "YATAI_T_ORG_UID"
ENV_YATAI_DEPLOYMENT_UID = "YATAI_T_DEPLOYMENT_UID"
ENV_YATAI_CLUSTER_UID = "YATAI_T_CLUSTER_UID"


@lru_cache(maxsize=1)
def get_platform() -> str:
    return platform(aliased=True)


@lru_cache(maxsize=1)
def get_python_version() -> str:
    return python_version()


@attr.define
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
            yaml.dump(attr.asdict(new_client_info), stream=f)

        return new_client_info


@inject
@lru_cache(maxsize=1)
def get_yatai_user_email(
    cloud_config: Path = Provide[BentoMLContainer.cloud_config],
) -> str | None:
    if cloud_config.exists():
        return CloudClientConfig.get_config(cloud_config).get_context().email


@lru_cache(maxsize=1)
def is_interactive() -> bool:
    import __main__ as main

    return not hasattr(main, "__file__")


@attr.define
class CommonProperties:
    # when the event is triggered
    timestamp: datetime = attr.field(factory=lambda: datetime.now(timezone.utc))

    # environment related
    platform: str = attr.field(factory=get_platform)
    bentoml_version: str = attr.field(default=BENTOML_VERSION)
    python_version: str = attr.field(factory=get_python_version)
    is_interactive: bool = attr.field(factory=is_interactive)
    in_notebook: bool = attr.field(factory=is_jupyter)

    # resource related
    memory_usage_percent: float = attr.field(init=False)
    total_memory_in_mb: float = attr.field(init=False)

    # client related
    client: ClientInfo = attr.field(factory=get_client_info)
    yatai_user_email: t.Optional[str] = attr.field(factory=get_yatai_user_email)
    yatai_version: t.Optional[str] = attr.field(
        default=os.environ.get(ENV_YATAI_VERSION, None)
    )
    yatai_org_uid: t.Optional[str] = attr.field(
        default=os.environ.get(ENV_YATAI_ORG_UID, None)
    )
    yatai_cluster_uid: t.Optional[str] = attr.field(
        default=os.environ.get(ENV_YATAI_CLUSTER_UID, None)
    )
    yatai_deployment_uid: t.Optional[str] = attr.field(
        default=os.environ.get(ENV_YATAI_DEPLOYMENT_UID, None)
    )

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


@attr.define
class CliEvent(EventMeta):
    cmd_group: str
    cmd_name: str
    duration_in_ms: float = attr.field(default=0)
    error_type: t.Optional[str] = attr.field(default=None)
    return_code: t.Optional[int] = attr.field(default=None)


@attr.define
class BentoBuildEvent(CliEvent):
    bento_creation_timestamp: t.Optional[datetime] = attr.field(default=None)
    bento_size_in_kb: float = attr.field(default=0)
    model_size_in_kb: float = attr.field(default=0)

    num_of_models: int = attr.field(default=0)
    num_of_runners: int = attr.field(default=0)
    model_types: t.List[str] = attr.field(factory=list)
    runnable_types: t.List[str] = attr.field(factory=list)


@attr.define
class ModelSaveEvent(EventMeta):
    module: str
    model_size_in_kb: float


# serve_kind determines different type of serving scenarios
SERVE_KIND = ["grpc", "http"]
# components are different components to be tracked.
COMPONENT_KIND = ["standalone", "api_server", "runner"]


@attr.define
class ServeInitEvent(EventMeta):
    serve_id: str
    production: bool
    serve_from_bento: bool
    serve_from_server_api: bool

    bento_creation_timestamp: t.Optional[datetime]
    serve_kind: str = attr.field(validator=attr.validators.in_(SERVE_KIND))
    num_of_models: int = attr.field(default=0)
    num_of_runners: int = attr.field(default=0)
    num_of_apis: int = attr.field(default=0)
    model_types: t.List[str] = attr.field(factory=list)
    runnable_types: t.List[str] = attr.field(factory=list)
    api_input_types: t.List[str] = attr.field(factory=list)
    api_output_types: t.List[str] = attr.field(factory=list)


@attr.define
class ServeUpdateEvent(EventMeta):
    serve_id: str
    production: bool
    triggered_at: datetime
    duration_in_seconds: int
    serve_kind: str = attr.field(validator=attr.validators.in_(SERVE_KIND))
    serve_from_server_api: bool
    component: str = attr.field(
        validator=attr.validators.and_(
            attr.validators.instance_of(str), attr.validators.in_(COMPONENT_KIND)
        )
    )
    metrics: t.List[t.Any] = attr.field(factory=list)


ALL_EVENT_TYPES = t.Union[
    CliEvent,
    ModelSaveEvent,
    BentoBuildEvent,
    ServeInitEvent,
    ServeUpdateEvent,
    EventMeta,
]


@attr.define
class TrackingPayload:
    session_id: str
    event_properties: ALL_EVENT_TYPES
    common_properties: CommonProperties
    event_type: str

    def to_dict(self):
        return bentoml_cattr.unstructure(self)
