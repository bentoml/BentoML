from __future__ import annotations

import typing as t

import attr

from bentoml._internal.cloud.schemas.modelschemas import AccessControl
from bentoml._internal.cloud.schemas.modelschemas import ApiServerBentoFunctionOverrides
from bentoml._internal.cloud.schemas.modelschemas import BentoConfigServiceSchema
from bentoml._internal.cloud.schemas.modelschemas import BentoRequestOverrides
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetConfigV2
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetHPAConf
from bentoml._internal.cloud.schemas.modelschemas import LabelItemSchema


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
class UpdateDeploymentSchema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
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
    bentoml_config_overrides: t.Optional[
        dict[str, BentoConfigServiceSchema]
    ] = attr.field(default=None)


@attr.define(kw_only=True)
class CreateDeploymentSchema(UpdateDeploymentSchema):
    __omit_if_default__ = True
    __forbid_extra_keys__ = True
    name: t.Optional[str] = attr.field(default=None)
    cluster: str
    distributed: t.Optional[bool] = attr.field(default=None)
