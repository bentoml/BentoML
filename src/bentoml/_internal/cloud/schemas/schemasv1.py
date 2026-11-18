from __future__ import annotations

from datetime import datetime

import attr

from bentoml._internal.cloud.schemas.modelschemas import BentoManifestSchema
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetCanaryRule
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetConfig
from bentoml._internal.cloud.schemas.modelschemas import LabelItemSchema
from bentoml._internal.cloud.schemas.modelschemas import ModelManifestSchema
from bentoml._internal.cloud.schemas.modelschemas import ResourceInstanceConfigSchema
from bentoml._internal.cloud.schemas.modelschemas import TransmissionStrategy


@attr.define
class BaseSchema:
    uid: str
    created_at: datetime
    updated_at: datetime | None
    deleted_at: datetime | None


@attr.define
class BaseListSchema:
    start: int
    count: int
    total: int


@attr.define
class ResourceSchema(BaseSchema):
    name: str
    resource_type: str
    labels: list[LabelItemSchema]


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
    items: list[OrganizationSchema]


@attr.define
class ClusterSchema(ResourceSchema):
    description: str
    organization_name: str
    creator: UserSchema
    is_first: bool | None = None


@attr.define
class ClusterConfigSchema:
    default_deployment_kube_namespace: str


@attr.define
class ClusterListSchema(BaseListSchema):
    items: list[ClusterSchema]


@attr.define
class CreateBentoRepositorySchema:
    name: str
    description: str


@attr.define
class BentoSchema(ResourceSchema):
    description: str
    version: str
    image_build_status: str
    upload_status: str
    upload_finished_reason: str
    presigned_upload_url: str
    presigned_download_url: str
    manifest: BentoManifestSchema | None = attr.field(default=None)
    transmission_strategy: TransmissionStrategy | None = attr.field(default=None)
    upload_id: str | None = attr.field(default=None)

    upload_started_at: datetime | None = attr.field(default=None)
    upload_finished_at: datetime | None = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class BentoRepositorySchema(ResourceSchema):
    description: str
    latest_bento: BentoSchema | None


@attr.define
class BentoWithRepositorySchema(BentoSchema):
    repository: BentoRepositorySchema = attr.field(default=None)


@attr.define
class BentoWithRepositoryListSchema(BaseListSchema):
    items: list[BentoWithRepositorySchema] = attr.field(factory=list)


@attr.define
class CreateBentoSchema:
    description: str
    version: str
    manifest: BentoManifestSchema | None = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)
    labels: list[LabelItemSchema] = attr.field(factory=list)


@attr.define
class UpdateBentoSchema:
    description: str | None = attr.field(default=None)
    manifest: BentoManifestSchema | None = attr.field(default=None)
    labels: list[LabelItemSchema] | None = attr.field(default=None)


@attr.define
class BentoFullSchema(BentoWithRepositorySchema):
    models: list[ModelWithRepositorySchema] = attr.field(factory=list)


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
    parts: list[CompletePartSchema]
    upload_id: str


@attr.define
class FinishUploadSchema:
    status: str | None
    reason: str | None


@attr.define
class CreateModelRepositorySchema:
    name: str
    description: str


@attr.define
class ModelSchema(ResourceSchema):
    description: str
    version: str
    image_build_status: str
    upload_status: str
    upload_finished_reason: str
    presigned_upload_url: str
    presigned_download_url: str
    manifest: ModelManifestSchema

    transmission_strategy: TransmissionStrategy | None = attr.field(default=None)
    upload_id: str | None = attr.field(default=None)

    upload_started_at: datetime | None = attr.field(default=None)
    upload_finished_at: datetime | None = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class ModelRepositorySchema(ResourceSchema):
    description: str
    latest_model: ModelSchema | None


@attr.define
class ModelWithRepositorySchema(ModelSchema):
    repository: ModelRepositorySchema = attr.field(default=None)


@attr.define
class ModelWithRepositoryListSchema(BaseListSchema):
    items: list[ModelWithRepositorySchema] = attr.field(factory=list)


@attr.define
class CreateModelSchema:
    description: str
    version: str
    manifest: ModelManifestSchema
    build_at: datetime = attr.field(factory=datetime.now)
    labels: list[LabelItemSchema] = attr.field(factory=list)


@attr.define
class BentoRepositoryListSchema(BaseListSchema):
    items: list[BentoRepositorySchema]


@attr.define
class BentoListSchema(BaseListSchema):
    items: list[BentoSchema]


@attr.define
class CreateDeploymentTargetSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    bento_repository: str
    bento: str
    config: DeploymentTargetConfig
    canary_rules: list[DeploymentTargetCanaryRule] | None = attr.field(default=None)


@attr.define
class DeploymentSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    creator: UserSchema
    cluster: ClusterSchema
    status: str
    kube_namespace: str
    latest_revision: DeploymentRevisionSchema | None = attr.field(
        default=None
    )  # Delete returns no latest revision


@attr.define
class DeploymentTargetSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    creator: UserSchema
    bento: BentoFullSchema
    config: DeploymentTargetConfig
    canary_rules: list[DeploymentTargetCanaryRule] | None = attr.field(default=None)


@attr.define
class DeploymentRevisionSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    creator: UserSchema
    status: str
    targets: list[DeploymentTargetSchema]


@attr.define
class ResourceInstanceSchema(ResourceSchema):
    display_name: str
    description: str
    config: ResourceInstanceConfigSchema


@attr.define(kw_only=True)
class ClusterFullSchema(ClusterSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    organization: OrganizationSchema
    kube_config: str
    config: ClusterConfigSchema
    grafana_root_path: str
    resource_instances: list[ResourceInstanceSchema]


@attr.define
class DeploymentListSchema(BaseListSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    items: list[DeploymentSchema]


@attr.define
class UpdateDeploymentSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    targets: list[CreateDeploymentTargetSchema]
    labels: list[LabelItemSchema] | None = attr.field(default=None)
    description: str | None = attr.field(default=None)
    do_not_deploy: bool | None = attr.field(default=None)


@attr.define(kw_only=True)
class CreateDeploymentSchema(UpdateDeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    name: str
    kube_namespace: str


@attr.define(kw_only=True)
class DeploymentFullSchema(DeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    urls: list[str]


@attr.define
class SecretItem:
    key: str
    sub_path: str | None = attr.field(default=None)
    value: str | None = attr.field(default=None)


@attr.define
class SecretContentSchema:
    type: str
    items: list[SecretItem]
    path: str | None = attr.field(default=None)
    # Secret availability stage: build-time only, runtime only, or both.
    stage: str | None = attr.field(default=None)


@attr.define
class SecretSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    description: str
    creator: UserSchema
    content: SecretContentSchema
    cluster: ClusterSchema


@attr.define
class SecretListSchema(BaseListSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    items: list[SecretSchema]


@attr.define
class CreateSecretSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    name: str
    content: SecretContentSchema
    description: str | None = attr.field(default=None)


@attr.define
class UpdateSecretSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    content: SecretContentSchema
    description: str | None = attr.field(default=None)


@attr.define
class ApiTokenSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    description: str
    scopes: list[str]
    user: UserSchema
    organization: OrganizationSchema
    expired_at: datetime | None = attr.field(default=None)
    last_used_at: datetime | None = attr.field(default=None)
    is_expired: bool = attr.field(default=False)
    is_api_token: bool = attr.field(default=True)
    is_organization_token: bool = attr.field(default=False)
    is_global_access: bool = attr.field(default=False)
    token: str | None = attr.field(default=None)  # Only returned on create


@attr.define
class ApiTokenListSchema(BaseListSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    items: list[ApiTokenSchema]


@attr.define
class CreateApiTokenSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    name: str
    description: str | None = attr.field(default=None)
    scopes: list[str] | None = attr.field(default=None)
    expired_at: datetime | None = attr.field(default=None)
