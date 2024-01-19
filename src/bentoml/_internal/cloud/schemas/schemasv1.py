from __future__ import annotations

import typing as t
from datetime import datetime

import attr

from bentoml._internal.cloud.schemas.modelschemas import BentoImageBuildStatus
from bentoml._internal.cloud.schemas.modelschemas import BentoManifestSchema
from bentoml._internal.cloud.schemas.modelschemas import BentoUploadStatus
from bentoml._internal.cloud.schemas.modelschemas import DeploymentMode
from bentoml._internal.cloud.schemas.modelschemas import DeploymentRevisionStatus
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStatus
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetCanaryRule
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetConfig
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetType
from bentoml._internal.cloud.schemas.modelschemas import LabelItemSchema
from bentoml._internal.cloud.schemas.modelschemas import ModelImageBuildStatus
from bentoml._internal.cloud.schemas.modelschemas import ModelManifestSchema
from bentoml._internal.cloud.schemas.modelschemas import ModelUploadStatus
from bentoml._internal.cloud.schemas.modelschemas import ResourceInstanceConfigSchema
from bentoml._internal.cloud.schemas.modelschemas import ResourceType
from bentoml._internal.cloud.schemas.modelschemas import TransmissionStrategy


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
    host_cluster_display_name: t.Optional[str]
    organization_name: str
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


@attr.define
class BentoSchema(ResourceSchema):
    description: str
    version: str
    image_build_status: BentoImageBuildStatus
    upload_status: BentoUploadStatus
    upload_finished_reason: str
    presigned_upload_url: str
    presigned_download_url: str
    manifest: t.Optional[BentoManifestSchema] = attr.field(default=None)
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
    manifest: t.Optional[BentoManifestSchema] = attr.field(default=None)
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


@attr.define
class DeploymentRevisionSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    creator: UserSchema
    status: DeploymentRevisionStatus
    targets: t.List[DeploymentTargetSchema]


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


@attr.define
class UpdateDeploymentSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    targets: t.List[CreateDeploymentTargetSchema]
    mode: t.Optional[DeploymentMode] = attr.field(default=None)
    labels: t.Optional[t.List[LabelItemSchema]] = attr.field(default=None)
    description: t.Optional[str] = attr.field(default=None)
    do_not_deploy: t.Optional[bool] = attr.field(default=None)


@attr.define(kw_only=True)
class CreateDeploymentSchema(UpdateDeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    name: str
    kube_namespace: str


@attr.define(kw_only=True)
class DeploymentFullSchema(DeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    urls: list[str]
