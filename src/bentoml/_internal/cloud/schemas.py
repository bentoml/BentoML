from __future__ import annotations

import json
import typing as t
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import attr
import cattr
from dateutil.parser import parse

time_format = "%Y-%m-%d %H:%M:%S.%f"

T = t.TypeVar("T")

if TYPE_CHECKING:
    from _bentoml_impl.config import ServiceConfig
else:
    ServiceConfig = t.Dict[str, t.Any]


def datetime_encoder(time_obj: t.Optional[datetime]) -> t.Optional[str]:
    if not time_obj:
        return None
    return time_obj.strftime(time_format)


def datetime_decoder(datetime_str: t.Optional[str], _: t.Any) -> t.Optional[datetime]:
    if not datetime_str:
        return None
    return parse(datetime_str)


def dict_options_converter(
    options_type: type[T],
) -> t.Callable[[T | dict[str, T]], T]:
    def _converter(value: T | dict[str, T] | None) -> T:
        if value is None:
            return options_type()
        if isinstance(value, dict):
            return options_type(**value)
        return value

    return _converter


cloud_converter = cattr.Converter()

cloud_converter.register_unstructure_hook(datetime, datetime_encoder)
cloud_converter.register_structure_hook(datetime, datetime_decoder)


def schema_from_json(json_content: str, cls: t.Type[T]) -> T:
    dct = json.loads(json_content)
    return cloud_converter.structure(dct, cls)


def schema_to_json(obj: t.Any) -> str:
    res = cloud_converter.unstructure(obj, obj.__class__)
    return json.dumps(res)


def schema_from_object(obj: t.Any, cls: t.Type[T]) -> T:
    return cloud_converter.structure(obj, cls)


@attr.define
class BaseSchema:
    uid: str
    created_at: datetime
    updated_at: t.Optional[datetime]
    deleted_at: t.Optional[datetime]


@attr.define
class BaseListSchema:
    start: int
    count: int
    total: int


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


@attr.define
class ResourceSchema(BaseSchema):
    name: str
    resource_type: ResourceType
    labels: t.List[LabelItemSchema]


@attr.define
class UserSchema:
    name: str
    email: str
    first_name: str
    last_name: str

    def get_name(self) -> str:
        if not self.first_name and not self.last_name:
            return self.name
        return f"{self.first_name} {self.last_name}".strip()


@attr.define
class OrganizationSchema(ResourceSchema):
    description: str


@attr.define
class OrganizationListSchema(BaseListSchema):
    items: t.List[OrganizationSchema]


@attr.define
class ClusterSchema(ResourceSchema):
    description: str
    creator: UserSchema


@attr.define
class ClusterConfigSchema:
    default_deployment_kube_namespace: str


@attr.define
class ClusterListSchema(BaseListSchema):
    items: t.List[ClusterSchema]


@attr.define
class CreateBentoRepositorySchema:
    name: str
    description: str


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
    name: str
    service: str
    bentoml_version: str
    size_bytes: int
    apis: t.Dict[str, BentoApiSchema] = attr.field(factory=dict)
    models: t.List[str] = attr.field(factory=list)
    runners: t.Optional[t.List[BentoRunnerSchema]] = attr.field(factory=list)
    config: ServiceConfig = attr.field(factory=dict)


if TYPE_CHECKING:
    TransmissionStrategy = t.Literal["presigned_url", "proxy"]
else:
    TransmissionStrategy = str


@attr.define
class BentoSchema(ResourceSchema):
    description: str
    version: str
    image_build_status: BentoImageBuildStatus
    upload_status: BentoUploadStatus
    upload_finished_reason: str
    presigned_upload_url: str
    presigned_download_url: str
    manifest: BentoManifestSchema
    transmission_strategy: t.Optional[TransmissionStrategy] = attr.field(default=None)
    upload_id: t.Optional[str] = attr.field(default=None)

    upload_started_at: t.Optional[datetime] = attr.field(default=None)
    upload_finished_at: t.Optional[datetime] = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class BentoRepositorySchema(ResourceSchema):
    description: str
    latest_bento: t.Optional[BentoSchema]


@attr.define
class BentoWithRepositorySchema(BentoSchema):
    repository: BentoRepositorySchema = attr.field(default=None)


@attr.define
class BentoWithRepositoryListSchema(BaseListSchema):
    items: t.List[BentoWithRepositorySchema] = attr.field(factory=list)


@attr.define
class CreateBentoSchema:
    description: str
    version: str
    manifest: BentoManifestSchema
    build_at: datetime = attr.field(factory=datetime.now)
    labels: t.List[LabelItemSchema] = attr.field(factory=list)


@attr.define
class UpdateBentoSchema:
    manifest: t.Optional[BentoManifestSchema] = attr.field(default=None)
    labels: t.Optional[t.List[LabelItemSchema]] = attr.field(default=None)


@attr.define
class BentoFullSchema(BentoWithRepositorySchema):
    models: t.List[ModelWithRepositorySchema] = attr.field(factory=list)


@attr.define
class PreSignMultipartUploadUrlSchema:
    upload_id: str
    part_number: int


@attr.define
class CompletePartSchema:
    part_number: int
    etag: str


@attr.define
class CompleteMultipartUploadSchema:
    parts: t.List[CompletePartSchema]
    upload_id: str


@attr.define
class FinishUploadBentoSchema:
    status: t.Optional[BentoUploadStatus]
    reason: t.Optional[str]


@attr.define
class CreateModelRepositorySchema:
    name: str
    description: str


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


@attr.define
class ModelSchema(ResourceSchema):
    description: str
    version: str
    image_build_status: ModelImageBuildStatus
    upload_status: ModelUploadStatus
    upload_finished_reason: str
    presigned_upload_url: str
    presigned_download_url: str
    manifest: ModelManifestSchema

    transmission_strategy: t.Optional[TransmissionStrategy] = attr.field(default=None)
    upload_id: t.Optional[str] = attr.field(default=None)

    upload_started_at: t.Optional[datetime] = attr.field(default=None)
    upload_finished_at: t.Optional[datetime] = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class ModelRepositorySchema(ResourceSchema):
    description: str
    latest_model: t.Optional[ModelSchema]


@attr.define
class ModelWithRepositorySchema(ModelSchema):
    repository: ModelRepositorySchema = attr.field(default=None)


@attr.define
class ModelWithRepositoryListSchema(BaseListSchema):
    items: t.List[ModelWithRepositorySchema] = attr.field(factory=list)


@attr.define
class CreateModelSchema:
    description: str
    version: str
    manifest: ModelManifestSchema
    build_at: datetime = attr.field(factory=datetime.now)
    labels: t.List[LabelItemSchema] = attr.field(factory=list)


@attr.define
class FinishUploadModelSchema:
    status: t.Optional[ModelUploadStatus]
    reason: t.Optional[str]


@attr.define
class BentoRepositoryListSchema(BaseListSchema):
    items: t.List[BentoRepositorySchema]


@attr.define
class BentoListSchema(BaseListSchema):
    items: t.List[BentoSchema]


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
    cpu: t.Optional[int] = attr.field(default=None)
    gpu: t.Optional[int] = attr.field(default=None)
    memory: t.Optional[str] = attr.field(default=None)
    qps: t.Optional[int] = attr.field(default=None)
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


@attr.define
class DeploymentTargetRunnerConfig:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    resource_instance: t.Optional[str] = attr.field(default=None)
    resources: t.Optional[DeploymentTargetResources] = attr.field(default=None)
    hpa_conf: t.Optional[DeploymentTargetHPAConf] = attr.field(default=None)
    envs: t.Optional[t.List[LabelItemSchema]] = attr.field(default=None)
    enable_stealing_traffic_debug_mode: t.Optional[bool] = attr.field(default=None)
    enable_debug_mode: t.Optional[bool] = attr.field(default=None)
    enable_debug_pod_receive_production_traffic: t.Optional[bool] = attr.field(
        default=None
    )
    deployment_strategy: t.Optional[DeploymentStrategy] = attr.field(default=None)
    bento_deployment_overrides: t.Optional[RunnerBentoDeploymentOverrides] = attr.field(
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
    resources: DeploymentTargetResources = attr.field(
        default=None, converter=dict_options_converter(DeploymentTargetResources)
    )
    kubeResourceUid: str = attr.field(default="")  # empty str
    kubeResourceVersion: str = attr.field(default="")
    resource_instance: t.Optional[str] = attr.field(default=None)
    hpa_conf: t.Optional[DeploymentTargetHPAConf] = attr.field(default=None)
    envs: t.Optional[t.List[LabelItemSchema]] = attr.field(default=None)
    runners: t.Optional[t.Dict[str, DeploymentTargetRunnerConfig]] = attr.field(
        default=None
    )
    access_control: t.Optional[str] = attr.field(default=None)
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
    traffic_control: t.Optional[TrafficControlConfig] = attr.field(default=None)
    deployment_cold_start_wait_timeout: t.Optional[int] = attr.field(default=None)


@attr.define
class CreateDeploymentTargetSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    type: DeploymentTargetType  # stable by default
    bento_repository: str
    bento: str
    config: DeploymentTargetConfig
    canary_rules: t.Optional[t.List[DeploymentTargetCanaryRule]] = attr.field(
        default=None
    )


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


@attr.define
class DeploymentSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    creator: UserSchema
    cluster: ClusterSchema
    status: DeploymentStatus
    kube_namespace: str
    latest_revision: t.Optional[DeploymentRevisionSchema] = attr.field(
        default=None
    )  # Delete returns no latest revision
    mode: t.Optional[DeploymentMode] = attr.field(default=None)


@attr.define
class DeploymentTargetSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    creator: UserSchema
    type: DeploymentTargetType
    bento: BentoFullSchema
    config: DeploymentTargetConfig
    canary_rules: t.Optional[t.List[DeploymentTargetCanaryRule]] = attr.field(
        default=None
    )


class DeploymentRevisionStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@attr.define
class DeploymentRevisionSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    creator: UserSchema
    status: DeploymentRevisionStatus
    targets: t.List[DeploymentTargetSchema]


@attr.define
class ResourceInstanceConfigSchema:
    group: str
    resources: DeploymentTargetResources
    price: str
    node_selectors: t.Optional[t.Dict[str, str]] = attr.field(factory=dict)


@attr.define
class ResourceInstanceSchema(ResourceSchema):
    display_name: str
    description: str
    config: ResourceInstanceConfigSchema


@attr.define
class ClusterFullSchema(ClusterSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    organization: OrganizationSchema
    kube_config: str
    config: ClusterConfigSchema
    grafana_root_path: str
    resource_instances: t.List[ResourceInstanceSchema]


@attr.define
class DeploymentListSchema(BaseListSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    items: t.List[DeploymentSchema]


class DeploymentMode(Enum):
    Deployment = "deployment"
    Function = "function"


@attr.define
class UpdateDeploymentSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    targets: t.List[CreateDeploymentTargetSchema]
    mode: t.Optional[DeploymentMode] = attr.field(default=None)
    labels: t.Optional[t.List[LabelItemSchema]] = attr.field(default=None)
    description: t.Optional[str] = attr.field(default=None)
    do_not_deploy: t.Optional[bool] = attr.field(default=None)


@attr.define
class CreateDeploymentSchema(UpdateDeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    name: str = attr.field(default=None)
    kube_namespace: t.Optional[str] = attr.field(default=None)


@attr.define
class FullDeploymentSchema(CreateDeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    cluster_name: t.Optional[str] = attr.field(default=None)
