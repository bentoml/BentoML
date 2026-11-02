from __future__ import annotations

import typing as t
from enum import Enum
from typing import TYPE_CHECKING

import attr

from ...bento.bento import BentoServiceInfo
from ...bento.bento import ImageInfo
from ...bento.build_config import BentoEnvSchema
from ...cloud.schemas.utils import dict_options_converter
from ...tag import Tag

time_format = "%Y-%m-%d %H:%M:%S.%f"

T = t.TypeVar("T")


class ResourceType(Enum):
    USER = "user"
    ORG = "organization"
    CLUSTER = "cluster"
    HostCluster = "host_cluster"
    BENTO_REPOSITORY = "bento_repository"
    BENTO = "bento"
    MODEL_REPOSITORY = "model_repository"
    MODEL = "model"
    DEPLOYMENT = "deployment"
    DEPLOYMENT_REVISION = "deployment_revision"
    TERMINAL_RECORD = "terminal_record"
    LABEL = "label"
    API_TOKEN = "api_token"
    YATAI_COMPONENT = "yatai_component"
    LimitGroup = "limit_group"
    ResourceInstance = "resource_instance"


class BentoImageBuildStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    SUCCESS = "success"
    FAILED = "failed"


class UploadStatus(Enum):
    PENDING = "pending"
    BUILDING = "uploading"
    SUCCESS = "success"
    FAILED = "failed"


@attr.define
class BentoApiSchema:
    route: str
    doc: str
    input: str
    output: str


@attr.define
class BentoRunnerResourceSchema:
    cpu: t.Any | None
    nvidia_gpu: t.Any | None
    custom_resources: t.Any | None


@attr.define
class BentoRunnerSchema:
    name: str
    runnable_type: str | None
    models: list[str] | None
    resource_config: BentoRunnerResourceSchema | None


@attr.define
class BentoManifestSchema:
    service: str
    bentoml_version: str = attr.field(eq=False)
    size_bytes: int = attr.field(eq=False)
    entry_service: str = ""
    name: str | None = None
    apis: dict[str, BentoApiSchema] = attr.field(factory=dict)
    models: list[str] = attr.field(factory=list, eq=False)
    runners: list[BentoRunnerSchema] | None = attr.field(factory=list)
    services: list[BentoServiceInfo] = attr.field(factory=list)
    envs: list[BentoEnvSchema] = attr.field(factory=list)
    schema: dict[str, t.Any] = attr.field(factory=dict)
    version: str | None = attr.field(default=None, eq=False)
    dev: bool = attr.field(default=False, eq=False)
    image: ImageInfo | None = attr.field(default=None, eq=False)
    spec: int = attr.field(default=1)

    @property
    def tag(self) -> Tag:
        return Tag(self.name, self.version)


if TYPE_CHECKING:
    TransmissionStrategy = t.Literal["presigned_url", "proxy"]
else:
    TransmissionStrategy = str


@attr.define
class ModelManifestSchema:
    module: str
    api_version: str
    bentoml_version: str
    size_bytes: int
    metadata: dict[str, t.Any] = attr.field(factory=dict)
    context: dict[str, t.Any] = attr.field(factory=dict)
    options: dict[str, t.Any] = attr.field(factory=dict)


@attr.define
class DeploymentTargetCanaryRule:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    type: str
    weight: int
    header: str
    cookie: str
    header_value: str


@attr.define
class ApiServerBentoDeploymentOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    monitorExporter: dict[str, t.Any] | None = attr.field(default=None)
    extraPodMetadata: dict[str, t.Any] | None = attr.field(default=None)
    extraPodSpec: dict[str, t.Any] | None = attr.field(default=None)


@attr.define
class ApiServerBentoFunctionOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    annotations: dict[str, str] | None = attr.field(default=None)
    monitorExporter: dict[str, t.Any] | None = attr.field(default=None)
    extraPodMetadata: dict[str, t.Any] | None = attr.field(default=None)
    extraPodSpec: dict[str, t.Any] | None = attr.field(default=None)


@attr.define
class RunnerBentoFunctionOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    annotations: dict[str, str] | None = attr.field(default=None)
    extraPodMetadata: dict[str, t.Any] | None = attr.field(default=None)
    extraPodSpec: dict[str, t.Any] | None = attr.field(default=None)


@attr.define
class RunnerBentoDeploymentOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    extraPodMetadata: dict[str, t.Any] | None = attr.field(default=None)
    extraPodSpec: dict[str, t.Any] | None = attr.field(default=None)


@attr.define
class BentoRequestOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    imageBuildTimeout: int = attr.field(default=None)
    imageBuilderExtraPodMetadata: dict[str, t.Any] | None = attr.field(default=None)
    imageBuilderExtraPodSpec: dict[str, t.Any] | None = attr.field(default=None)
    imageBuilderExtraContainerEnv: list[dict[str, t.Any]] | None = attr.field(
        default=None
    )
    imageBuilderContainerResources: dict[str, t.Any] | None = attr.field(default=None)
    dockerConfigJsonSecretName: str | None = attr.field(default=None)
    downloaderContainerEnvFrom: dict[str, t.Any] | None = attr.field(default=None)


@attr.define
class LabelItemSchema:
    key: str
    value: str


@attr.define
class EnvItemSchema:
    name: str
    value: str


@attr.define
class HPAMetric:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    type: str
    value: t.Any  # resource.Quantity


@attr.define
class HPAPolicy:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    metrics: list[HPAMetric] | None = attr.field(default=None)
    scale_down_behavior: str | None = attr.field(default=None)
    scale_up_behavior: str | None = attr.field(default=None)
    scale_down_stabilization_window: int | None = attr.field(default=None)
    scale_up_stabilization_window: int | None = attr.field(default=None)


@attr.define
class DeploymentTargetHPAConf:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    min_replicas: int | None = attr.field(default=None)
    max_replicas: int | None = attr.field(default=None)
    policy: HPAPolicy | None = attr.field(default=None)


@attr.define
class DeploymentTargetResourceItem:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    cpu: str | None = attr.field(default=None)
    memory: str | None = attr.field(default=None)
    gpu: str | None = attr.field(default=None)
    custom: dict[str, str] | None = attr.field(default=None)


@attr.define
class DeploymentTargetResources:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    requests: DeploymentTargetResourceItem | None = attr.field(default=None)
    limits: DeploymentTargetResourceItem | None = attr.field(default=None)


@attr.define
class RequestQueueConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    enabled: bool | None = attr.field(default=None)
    max_consume_concurrency: int | None = attr.field(default=None)


@attr.define
class TrafficControlConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    timeout: str | None = attr.field(default=None)
    request_queue: RequestQueueConfig | None = attr.field(default=None)


class DeploymentStrategy(Enum):
    RollingUpdate = "RollingUpdate"
    Recreate = "Recreate"
    RampedSlowRollout = "RampedSlowRollout"
    BestEffortControlledRollout = "BestEffortControlledRollout"


@attr.define
class DeploymentTargetRunnerConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    resource_instance: str | None = attr.field(default=None)
    resources: DeploymentTargetResources | None = attr.field(default=None)
    hpa_conf: DeploymentTargetHPAConf | None = attr.field(default=None)
    envs: list[LabelItemSchema | None] | None = attr.field(default=None)
    enable_stealing_traffic_debug_mode: bool | None = attr.field(default=None)
    enable_debug_mode: bool | None = attr.field(default=None)
    enable_debug_pod_receive_production_traffic: bool | None = attr.field(default=None)
    deployment_strategy: str | None = attr.field(default=None)
    bento_deployment_overrides: RunnerBentoDeploymentOverrides | None = attr.field(
        default=None
    )
    bento_function_overrides: RunnerBentoFunctionOverrides | None = attr.field(
        default=None
    )
    traffic_control: TrafficControlConfig | None = attr.field(default=None)
    deployment_cold_start_wait_timeout: int | None = attr.field(default=None)


@attr.define
class DeploymentTargetConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    resources: DeploymentTargetResources | None = attr.field(
        default=None, converter=dict_options_converter(DeploymentTargetResources)
    )
    kubeResourceUid: str = attr.field(default="")  # empty str
    kubeResourceVersion: str = attr.field(default="")
    resource_instance: str | None = attr.field(default=None)
    hpa_conf: DeploymentTargetHPAConf | None = attr.field(default=None)
    envs: list[LabelItemSchema | None] | None = attr.field(default=None)
    runners: dict[str, DeploymentTargetRunnerConfig] | None = attr.field(default=None)
    access_control: str | None = attr.field(default=None)
    enable_ingress: bool | None = attr.field(default=None)  # false for enables
    enable_stealing_traffic_debug_mode: bool | None = attr.field(default=None)
    enable_debug_mode: bool | None = attr.field(default=None)
    enable_debug_pod_receive_production_traffic: bool | None = attr.field(default=None)
    deployment_strategy: str | None = attr.field(default=None)  # Specific
    bento_deployment_overrides: ApiServerBentoDeploymentOverrides | None = attr.field(
        default=None
    )
    bento_request_overrides: BentoRequestOverrides | None = attr.field(
        default=None
    )  # Put into image builder
    bento_function_overrides: ApiServerBentoFunctionOverrides | None = attr.field(
        default=None
    )
    traffic_control: TrafficControlConfig | None = attr.field(default=None)
    deployment_cold_start_wait_timeout: int | None = attr.field(default=None)


@attr.define
class ExtraDeploymentOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    bento_function_overrides: ApiServerBentoFunctionOverrides | None = attr.field(
        default=None
    )
    bento_request_overrides: BentoRequestOverrides | None = attr.field(default=None)


@attr.define
class DeploymentServiceConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    instance_type: str | None = attr.field(default=None)
    scaling: DeploymentTargetHPAConf | None = attr.field(default=None)
    envs: list[EnvItemSchema | None] | None = attr.field(default=None)
    deployment_strategy: str | None = attr.field(default=None)
    extras: ExtraDeploymentOverrides | None = attr.field(default=None)
    cold_start_timeout: int | None = attr.field(default=None)
    config_overrides: dict[str, t.Any] | None = attr.field(factory=dict)


class DeploymentStatus(Enum):
    Unknown = "unknown"
    NonDeployed = "non-deployed"
    Running = "running"
    Unhealthy = "unhealthy"
    Failed = "failed"
    Deploying = "deploying"
    Terminating = "terminating"
    Terminated = "terminated"
    ImageBuilding = "image-building"
    ImageBuildFailed = "image-build-failed"
    ImageBuildSucceeded = "image-build-succeeded"
    ScaledToZero = "scaled-to-zero"


@attr.define
class ResourceInstanceConfigSchema:
    group: str
    resources: DeploymentTargetResources
    price: str
    node_selectors: dict[str, str] | None = attr.field(factory=dict)
    gpu_config: ResourceInstanceGPUConfigSchema | None = attr.field(default=None)


@attr.define
class ResourceInstanceGPUConfigSchema:
    type: str
    memory: str


class DeploymentRevisionStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
