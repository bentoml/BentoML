from __future__ import annotations

import typing as t

import attr

from bentoml._internal.cloud.schemas.modelschemas import DeploymentServiceConfig
from bentoml._internal.cloud.schemas.schemasv1 import BaseListSchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoWithRepositorySchema
from bentoml._internal.cloud.schemas.schemasv1 import ClusterSchema
from bentoml._internal.cloud.schemas.schemasv1 import ResourceSchema
from bentoml._internal.cloud.schemas.schemasv1 import UserSchema

from .modelschemas import EnvItemSchema
from .modelschemas import LabelItemSchema


@attr.define
class DeploymentTargetSchema(ResourceSchema):
    creator: UserSchema | None
    config: DeploymentConfigSchema | None
    bento: BentoWithRepositorySchema | None
    kube_resource_uid: str | None = attr.field(default=None)
    kube_resource_version: str | None = attr.field(default=None)


@attr.define
class DeploymentTargetsSchema:
    main: DeploymentTargetSchema
    canary: DeploymentCanarySchema | None = attr.field(default=None)


@attr.define
class DeploymentTargetListSchema(BaseListSchema):
    items: list[DeploymentTargetSchema | None]


@attr.define
class DeploymentRevisionSchema(ResourceSchema):
    creator: UserSchema | None
    status: str
    targets: list[DeploymentTargetSchema | None]


@attr.define
class DeploymentRevisionListSchema(BaseListSchema):
    items: list[DeploymentRevisionSchema | None]


@attr.define
class DeploymentConfigSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    access_authorization: bool = attr.field(default=False)
    envs: list[EnvItemSchema] | None = attr.field(default=None)
    labels: list[LabelItemSchema] | None = attr.field(default=None)
    secrets: list[str] | None = attr.field(default=None)
    services: dict[str, DeploymentServiceConfig] = attr.field(factory=dict)
    canary: DeploymentCanarySchema | None = attr.field(default=None)


@attr.define
class DeploymentCanaryTargetSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    envs: list[EnvItemSchema] | None = attr.field(default=None)
    secrets: list[str] | None = attr.field(default=None)
    services: dict[str, DeploymentServiceConfig] = attr.field(factory=dict)


@attr.define
class DeploymentVersionSchema(DeploymentCanaryTargetSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    bento: str | None = attr.field(default=None)
    weight: int | None = attr.field(default=None)


@attr.define
class DeploymentRoutingSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    route_type: t.Literal["random", "header", "query"] | None = attr.field(default=None)
    route_by: str | None = attr.field(default=None)


@attr.define
class DeploymentCanarySchema(DeploymentRoutingSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    versions: dict[str, DeploymentVersionSchema] | None = attr.field(default=None)


@attr.define(kw_only=True)
class UpdateDeploymentSchema(DeploymentConfigSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False  # distributed, cluster and name need to be ignored
    bento: str


@attr.define(kw_only=True)
class CreateDeploymentSchema(UpdateDeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    name: str | None = None
    dev: bool = False


@attr.define
class DeploymentRoutingManifestSchema(DeploymentRoutingSchema):
    __forbid_extra_keys__ = False
    __omit_if_default__ = True
    weights: dict[str, int] | None = None


@attr.define
class DeploymentManifestSchema:
    __forbid_extra_keys__ = False
    dev: bool = False
    routing: DeploymentRoutingManifestSchema | None = None


@attr.define
class DeploymentSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    status: str
    kube_namespace: str
    creator: UserSchema
    cluster: ClusterSchema
    latest_revision: DeploymentRevisionSchema | None
    manifest: DeploymentManifestSchema | None = None


@attr.define
class DeploymentFullSchema(DeploymentSchema):
    urls: list[str] = attr.field(factory=list)


@attr.define
class DeploymentListSchema(BaseListSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    items: list[DeploymentSchema]


@attr.define
class KubePodStatusSchema:
    __forbid_extra_keys__ = False
    status: str
    reason: str


@attr.define
class PodStatusSchema:
    __forbid_extra_keys__ = False
    phase: str
    ready: bool


@attr.define
class KubePodSchema:
    __forbid_extra_keys__ = False
    name: str
    namespace: str
    labels: dict[str, str]
    pod_status: KubePodStatusSchema
    status: PodStatusSchema
    runner_name: str


@attr.define
class LogSchema:
    __forbid_extra_keys__ = False
    items: list[str] = attr.field(factory=list)
    type: str = "append"


@attr.define
class LogWSResponseSchema:
    __forbid_extra_keys__ = False
    message: str | None
    type: str
    payload: LogSchema | None


@attr.define
class KubePodWSResponseSchema:
    __forbid_extra_keys__ = False
    message: str
    type: str
    payload: list[KubePodSchema] | None


@attr.define
class UploadDeploymentFileSchema:
    __forbid_extra_keys__ = False
    path: str
    b64_encoded_content: str


@attr.define
class UploadDeploymentFilesSchema:
    __forbid_extra_keys__ = False
    files: list[UploadDeploymentFileSchema]


@attr.define
class DeleteDeploymentFilesSchema:
    __forbid_extra_keys__ = False
    paths: list[str]


@attr.define
class DeploymentFileSchema:
    __forbid_extra_keys__ = False
    path: str
    size: int
    md5: str


@attr.define
class DeploymentFileListSchema:
    __forbid_extra_keys__ = False
    files: list[DeploymentFileSchema]
