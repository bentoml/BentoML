"""
User facing python APIs for deployment
"""

from __future__ import annotations

import typing as t

import attr
from simple_di import Provide

from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.cloud.deployment import DeploymentInfo
from bentoml._internal.cloud.schemas.modelschemas import EnvItemSchema
from bentoml._internal.tag import Tag
from bentoml.exceptions import BentoMLException

from ._internal.configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from ._internal.bento import BentoStore


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    *,
    bento: Tag | str | None = ...,
    cluster: str | None = ...,
    access_authorization: bool | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[EnvItemSchema] | t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> DeploymentInfo: ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    *,
    bento: Tag | str | None = ...,
    config_file: str | None = ...,
) -> DeploymentInfo: ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    *,
    bento: Tag | str | None = ...,
    config_dict: dict[str, t.Any] | None = ...,
) -> DeploymentInfo: ...


def create(
    name: str | None = None,
    path_context: str | None = None,
    *,
    bento: Tag | str | None = None,
    cluster: str | None = None,
    access_authorization: bool | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: t.List[EnvItemSchema] | t.List[dict[str, t.Any]] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> DeploymentInfo:
    config_params = DeploymentConfigParameters(
        name=name,
        path_context=path_context,
        bento=bento,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=(
            [
                attr.asdict(item) if isinstance(item, EnvItemSchema) else item
                for item in envs
            ]
            if envs is not None
            else None
        ),
        extras=extras,
        config_dict=config_dict,
        config_file=config_file,
    )
    try:
        config_params.verify()
    except BentoMLException as e:
        raise BentoMLException(
            f"Failed to create deployment due to invalid configuration: {e}"
        )
    return Deployment.create(deployment_config_params=config_params)


@t.overload
def update(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = ...,
    access_authorization: bool | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[EnvItemSchema] | t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> DeploymentInfo: ...


@t.overload
def update(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = ...,
    config_file: str | None = ...,
) -> DeploymentInfo: ...


@t.overload
def update(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = ...,
    config_dict: dict[str, t.Any] | None = ...,
) -> DeploymentInfo: ...


def update(
    name: str | None = None,
    path_context: str | None = None,
    cluster: str | None = None,
    *,
    bento: Tag | str | None = None,
    access_authorization: bool | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: (
        t.List[EnvItemSchema]
        | t.List[dict[str, t.Any]]
        | t.List[dict[str, t.Any]]
        | None
    ) = None,
    extras: dict[str, t.Any] | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> DeploymentInfo:
    config_params = DeploymentConfigParameters(
        name=name,
        path_context=path_context,
        bento=bento,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=(
            [
                attr.asdict(item) if isinstance(item, EnvItemSchema) else item
                for item in envs
            ]
            if envs is not None
            else None
        ),
        extras=extras,
        config_dict=config_dict,
        config_file=config_file,
    )
    try:
        config_params.verify()
    except BentoMLException as e:
        raise BentoMLException(
            f"Failed to create deployment due to invalid configuration: {e}"
        )

    return Deployment.update(deployment_config_params=config_params)


@t.overload
def apply(
    name: str | None = ...,
    cluster: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_dict: t.Optional[dict[str, t.Any]] = ...,
) -> DeploymentInfo: ...


@t.overload
def apply(
    name: str | None = ...,
    cluster: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_file: t.Optional[str] = ...,
) -> DeploymentInfo: ...


def apply(
    name: str | None = None,
    cluster: str | None = None,
    path_context: str | None = None,
    *,
    bento: Tag | str | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> DeploymentInfo:
    config_params = DeploymentConfigParameters(
        name=name,
        path_context=path_context,
        bento=bento,
        cluster=cluster,
        config_dict=config_dict,
        config_file=config_file,
    )
    try:
        config_params.verify()
    except BentoMLException as e:
        raise BentoMLException(
            f"Failed to create deployment due to invalid configuration: {e}"
        )

    return Deployment.apply(
        deployment_config_params=config_params,
    )


def get(
    name: str,
    cluster: str | None = None,
) -> DeploymentInfo:
    return Deployment.get(name=name, cluster=cluster)


def terminate(
    name: str,
    cluster: str | None = None,
) -> DeploymentInfo:
    return Deployment.terminate(name=name, cluster=cluster)


def delete(
    name: str,
    cluster: str | None = None,
) -> None:
    Deployment.delete(name=name, cluster=cluster)


def list(
    cluster: str | None = None,
    search: str | None = None,
) -> t.List[DeploymentInfo]:
    return Deployment.list(cluster=cluster, search=search)


__all__ = ["create", "get", "update", "apply", "terminate", "delete", "list"]
