from __future__ import annotations

import typing as t

import attr

from bentoml._internal.cloud.schemas.modelschemas import DeploymentMode
from bentoml._internal.cloud.schemas.modelschemas import DeploymentRevisionStatus
from bentoml._internal.cloud.schemas.modelschemas import DeploymentServiceConfig
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStatus
from bentoml._internal.cloud.schemas.schemasv1 import BaseListSchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoWithRepositorySchema
from bentoml._internal.cloud.schemas.schemasv1 import ClusterSchema
from bentoml._internal.cloud.schemas.schemasv1 import ResourceSchema
from bentoml._internal.cloud.schemas.schemasv1 import UserSchema

from .modelschemas import EnvItemSchema


@attr.define
class DeploymentTargetSchema(ResourceSchema):
    creator: t.Optional[UserSchema]
    config: t.Optional[DeploymentConfigSchema]
    bento: t.Optional[BentoWithRepositorySchema]
    kube_resource_uid: t.Optional[str] = attr.field(default=None)
    kube_resource_version: t.Optional[str] = attr.field(default=None)


@attr.define
class DeploymentTargetListSchema(BaseListSchema):
    items: t.List[t.Optional[DeploymentTargetSchema]]


@attr.define
class DeploymentRevisionSchema(ResourceSchema):
    creator: t.Optional[UserSchema]
    status: DeploymentRevisionStatus
    targets: t.List[t.Optional[DeploymentTargetSchema]]


@attr.define
class DeploymentRevisionListSchema(BaseListSchema):
    items: t.List[t.Optional[DeploymentRevisionSchema]]


@attr.define
class DeploymentConfigSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    access_authorization: bool = attr.field(default=False)
    envs: t.Optional[t.List[EnvItemSchema]] = attr.field(default=None)
    services: t.Dict[str, DeploymentServiceConfig] = attr.field(factory=dict)


@attr.define(kw_only=True)
class UpdateDeploymentSchema(DeploymentConfigSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = False  # distributed, cluster and name need to be ignored
    bento: str


@attr.define(kw_only=True)
class CreateDeploymentSchema(UpdateDeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    name: t.Optional[str] = attr.field(default=None)


@attr.define
class DeploymentSchema(ResourceSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    status: DeploymentStatus
    kube_namespace: str
    creator: UserSchema
    cluster: ClusterSchema
    latest_revision: t.Optional[DeploymentRevisionSchema]
    mode: t.Optional[DeploymentMode] = attr.field(default=None)


@attr.define
class DeploymentFullSchema(DeploymentSchema):
    urls: t.List[str] = attr.field(factory=list)


@attr.define
class DeploymentListSchema(BaseListSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    items: t.List[DeploymentSchema]
