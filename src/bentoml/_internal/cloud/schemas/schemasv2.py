from __future__ import annotations

import typing as t

import attr

from bentoml._internal.cloud.schemas.modelschemas import AccessControl
from bentoml._internal.cloud.schemas.modelschemas import ApiServerBentoFunctionOverrides
from bentoml._internal.cloud.schemas.modelschemas import BentoConfigManifestSchema
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetConfigV2
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetHPAConf
from bentoml._internal.cloud.schemas.modelschemas import LabelItemSchema


@attr.define
class ExtraDeploymentOverrides:
    bento_function_overrides: t.Optional[ApiServerBentoFunctionOverrides] = attr.field(
        default=None
    )
    bento_config_overrides: t.Optional[BentoConfigManifestSchema] = attr.field(
        default=None
    )


@attr.define
class UpdateDeploymentSchemaV2:
    bento: str
    description: t.Optional[str] = attr.field(default=None)
    access_type: t.Optional[AccessControl] = attr.field(default=None)
    instance_type: t.Optional[str] = attr.field(default=None)
    scaling: t.Optional[DeploymentTargetHPAConf] = attr.field(default=None)
    envs: t.Optional[t.List[t.Optional[LabelItemSchema]]] = attr.field(default=None)
    services: t.Optional[t.Dict[str, DeploymentTargetConfigV2]] = attr.field(
        factory=dict
    )
    deployment_strategy: t.Optional[DeploymentStrategy] = attr.field(default=None)
    extras: t.Optional[ExtraDeploymentOverrides] = attr.field(default=None)


@attr.define(kw_only=True)
class CreateDeploymentSchema(UpdateDeploymentSchemaV2):
    name: t.Optional[str] = attr.field(default=None)
    cluster: str
    distributed: t.Optional[bool] = attr.field(default=None)
