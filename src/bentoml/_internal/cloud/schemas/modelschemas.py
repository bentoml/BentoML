from __future__ import annotations

import typing as t
from enum import Enum
from typing import TYPE_CHECKING

import attr

from bentoml._internal.cloud.schemas.utils import dict_options_converter

from ...bento.bento import BentoServiceInfo

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


class BentoUploadStatus(Enum):
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
    cpu: t.Optional[t.Any]
    nvidia_gpu: t.Optional[t.Any]
    custom_resources: t.Optional[t.Any]


@attr.define
class BentoRunnerSchema:
    name: str
    runnable_type: t.Optional[str]
    models: t.Optional[t.List[str]]
    resource_config: t.Optional[BentoRunnerResourceSchema]


@attr.define
class BentoManifestSchema:
    service: str
    bentoml_version: str
    size_bytes: int
    entry_service: str = attr.field(default="")
    name: t.Optional[str] = attr.field(default=None)
    apis: t.Dict[str, BentoApiSchema] = attr.field(factory=dict)
    models: t.List[str] = attr.field(factory=list)
    runners: t.Optional[t.List[BentoRunnerSchema]] = attr.field(factory=list)
    services: t.List[BentoServiceInfo] = attr.field(factory=dict)
    envs: t.List[t.Dict[str, str]] = attr.field(factory=list)
    schema: t.Dict[str, t.Any] = attr.field(factory=dict)


if TYPE_CHECKING:
    TransmissionStrategy = t.Literal["presigned_url", "proxy"]
else:
    TransmissionStrategy = str


class ModelImageBuildStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    SUCCESS = "success"
    FAILED = "failed"


class ModelUploadStatus(Enum):
    PENDING = "pending"
    BUILDING = "uploading"
    SUCCESS = "success"
    FAILED = "failed"


@attr.define
class ModelManifestSchema:
    module: str
    api_version: str
    bentoml_version: str
    size_bytes: int
    metadata: t.Dict[str, t.Any] = attr.field(factory=dict)
    context: t.Dict[str, t.Any] = attr.field(factory=dict)
    options: t.Dict[str, t.Any] = attr.field(factory=dict)


class DeploymentTargetCanaryRuleType(Enum):
    WEIGHT = "weight"
    HEADER = "header"
    COOKIE = "cookie"


@attr.define
class DeploymentTargetCanaryRule:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    type: DeploymentTargetCanaryRuleType
    weight: int
    header: str
    cookie: str
    header_value: str


@attr.define
class ApiServerBentoDeploymentOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    monitorExporter: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
    extraPodMetadata: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
    extraPodSpec: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)


@attr.define
class ApiServerBentoFunctionOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    annotations: t.Optional[t.Dict[str, str]] = attr.field(default=None)
    monitorExporter: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
    extraPodMetadata: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
    extraPodSpec: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)


@attr.define
class RunnerBentoFunctionOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    annotations: t.Optional[t.Dict[str, str]] = attr.field(default=None)
    extraPodMetadata: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
    extraPodSpec: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)


@attr.define
class RunnerBentoDeploymentOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    extraPodMetadata: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
    extraPodSpec: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)


@attr.define
class BentoRequestOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    imageBuildTimeout: int = attr.field(default=None)
    imageBuilderExtraPodMetadata: t.Optional[t.Dict[str, t.Any]] = attr.field(
        default=None
    )
    imageBuilderExtraPodSpec: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
    imageBuilderExtraContainerEnv: t.Optional[t.List[t.Dict[str, t.Any]]] = attr.field(
        default=None
    )
    imageBuilderContainerResources: t.Optional[t.Dict[str, t.Any]] = attr.field(
        default=None
    )
    dockerConfigJsonSecretName: t.Optional[str] = attr.field(default=None)
    downloaderContainerEnvFrom: t.Optional[t.Dict[str, t.Any]] = attr.field(
        default=None
    )


@attr.define
class LabelItemSchema:
    key: str
    value: str


@attr.define
class EnvItemSchema:
    name: str
    value: str


class HPAMetricType(Enum):
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    QPS = "qps"


@attr.define
class HPAMetric:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    type: HPAMetricType  # enum
    value: t.Any  # resource.Quantity


class HPAScaleBehavior(Enum):
    DISABLED = "disabled"
    STABLE = "stable"
    FAST = "fast"


@attr.define
class HPAPolicy:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    metrics: t.Optional[t.List[HPAMetric]] = attr.field(default=None)
    scale_down_behavior: t.Optional[HPAScaleBehavior] = attr.field(default=None)
    scale_up_behavior: t.Optional[HPAScaleBehavior] = attr.field(default=None)


@attr.define
class DeploymentTargetHPAConf:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    min_replicas: t.Optional[int] = attr.field(default=None)
    max_replicas: t.Optional[int] = attr.field(default=None)
    policy: t.Optional[HPAPolicy] = attr.field(default=None)


@attr.define
class DeploymentTargetResourceItem:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    cpu: t.Optional[str] = attr.field(default=None)
    memory: t.Optional[str] = attr.field(default=None)
    gpu: t.Optional[str] = attr.field(default=None)
    custom: t.Optional[t.Dict[str, str]] = attr.field(default=None)


@attr.define
class DeploymentTargetResources:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    requests: t.Optional[DeploymentTargetResourceItem] = attr.field(default=None)
    limits: t.Optional[DeploymentTargetResourceItem] = attr.field(default=None)


@attr.define
class RequestQueueConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    enabled: t.Optional[bool] = attr.field(default=None)
    max_consume_concurrency: t.Optional[int] = attr.field(default=None)


@attr.define
class TrafficControlConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    timeout: t.Optional[str] = attr.field(default=None)
    request_queue: t.Optional[RequestQueueConfig] = attr.field(default=None)


class DeploymentStrategy(Enum):
    RollingUpdate = "RollingUpdate"
    Recreate = "Recreate"
    RampedSlowRollout = "RampedSlowRollout"
    BestEffortControlledRollout = "BestEffortControlledRollout"


class AccessControl(Enum):
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


@attr.define
class DeploymentTargetRunnerConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    resource_instance: t.Optional[str] = attr.field(default=None)
    resources: t.Optional[DeploymentTargetResources] = attr.field(default=None)
    hpa_conf: t.Optional[DeploymentTargetHPAConf] = attr.field(default=None)
    envs: t.Optional[t.List[t.Optional[LabelItemSchema]]] = attr.field(default=None)
    enable_stealing_traffic_debug_mode: t.Optional[bool] = attr.field(default=None)
    enable_debug_mode: t.Optional[bool] = attr.field(default=None)
    enable_debug_pod_receive_production_traffic: t.Optional[bool] = attr.field(
        default=None
    )
    deployment_strategy: t.Optional[DeploymentStrategy] = attr.field(default=None)
    bento_deployment_overrides: t.Optional[RunnerBentoDeploymentOverrides] = attr.field(
        default=None
    )
    bento_function_overrides: t.Optional[RunnerBentoFunctionOverrides] = attr.field(
        default=None
    )
    traffic_control: t.Optional[TrafficControlConfig] = attr.field(default=None)
    deployment_cold_start_wait_timeout: t.Optional[int] = attr.field(default=None)


class DeploymentTargetType(Enum):
    STABLE = "stable"
    CANARY = "canary"


@attr.define
class DeploymentTargetConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    resources: t.Optional[DeploymentTargetResources] = attr.field(
        default=None, converter=dict_options_converter(DeploymentTargetResources)
    )
    kubeResourceUid: str = attr.field(default="")  # empty str
    kubeResourceVersion: str = attr.field(default="")
    resource_instance: t.Optional[str] = attr.field(default=None)
    hpa_conf: t.Optional[DeploymentTargetHPAConf] = attr.field(default=None)
    envs: t.Optional[t.List[t.Optional[LabelItemSchema]]] = attr.field(default=None)
    runners: t.Optional[t.Dict[str, DeploymentTargetRunnerConfig]] = attr.field(
        default=None
    )
    access_control: t.Optional[AccessControl] = attr.field(default=None)
    enable_ingress: t.Optional[bool] = attr.field(default=None)  # false for enables
    enable_stealing_traffic_debug_mode: t.Optional[bool] = attr.field(default=None)
    enable_debug_mode: t.Optional[bool] = attr.field(default=None)
    enable_debug_pod_receive_production_traffic: t.Optional[bool] = attr.field(
        default=None
    )
    deployment_strategy: t.Optional[DeploymentStrategy] = attr.field(
        default=None
    )  # Specific
    bento_deployment_overrides: t.Optional[
        ApiServerBentoDeploymentOverrides
    ] = attr.field(default=None)
    bento_request_overrides: t.Optional[BentoRequestOverrides] = attr.field(
        default=None
    )  # Put into image builder
    bento_function_overrides: t.Optional[ApiServerBentoFunctionOverrides] = attr.field(
        default=None
    )
    traffic_control: t.Optional[TrafficControlConfig] = attr.field(default=None)
    deployment_cold_start_wait_timeout: t.Optional[int] = attr.field(default=None)


@attr.define
class ExtraDeploymentOverrides:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    bento_function_overrides: t.Optional[ApiServerBentoFunctionOverrides] = attr.field(
        default=None
    )
    bento_request_overrides: t.Optional[BentoRequestOverrides] = attr.field(
        default=None
    )


@attr.define
class DeploymentServiceConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    instance_type: t.Optional[str] = attr.field(default=None)
    scaling: t.Optional[DeploymentTargetHPAConf] = attr.field(default=None)
    envs: t.Optional[t.List[t.Optional[EnvItemSchema]]] = attr.field(default=None)
    deployment_strategy: t.Optional[DeploymentStrategy] = attr.field(default=None)
    extras: t.Optional[ExtraDeploymentOverrides] = attr.field(default=None)
    cold_start_timeout: t.Optional[int] = attr.field(default=None)
    config_overrides: t.Optional[t.Dict[str, t.Any]] = attr.field(factory=dict)


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


class DeploymentMode(Enum):
    Deployment = "deployment"
    Function = "function"


@attr.define
class ResourceInstanceConfigSchema:
    group: str
    resources: DeploymentTargetResources
    price: str
    node_selectors: t.Optional[t.Dict[str, str]] = attr.field(factory=dict)
    gpu_config: t.Optional[ResourceInstanceGPUConfigSchema] = attr.field(default=None)


@attr.define
class ResourceInstanceGPUConfigSchema:
    type: str
    memory: str


class DeploymentRevisionStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
