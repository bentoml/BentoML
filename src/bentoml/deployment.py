"""
User facing python APIs for deployment
"""

from __future__ import annotations

import typing as t

from simple_di import Provide
from simple_di import inject

from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.deployment import DeploymentInfo
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.tag import Tag
from bentoml.exceptions import BentoMLException

from ._internal.configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from ._internal.bento import BentoStore


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = ...,
    cluster: str | None = ...,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> DeploymentInfo:
    ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = ...,
    config_file: str | None = ...,
) -> DeploymentInfo:
    ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = ...,
    config_dict: dict[str, t.Any] | None = ...,
) -> DeploymentInfo:
    ...


@inject
def create(
    name: str | None = None,
    path_context: str | None = None,
    context: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = None,
    cluster: str | None = None,
    access_type: str | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: t.List[dict[str, t.Any]] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> DeploymentInfo:
    config_params = DeploymentConfigParameters(
        name=name,
        path_context=path_context,
        context=context,
        bento=bento,
        cluster=cluster,
        access_type=access_type,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=envs,
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
    return Deployment.create(
        deployment_config_params=config_params,
        context=context,
    )


@t.overload
def update(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = ...,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> DeploymentInfo:
    ...


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
) -> DeploymentInfo:
    ...


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
) -> DeploymentInfo:
    ...


@inject
def update(
    name: str | None = None,
    path_context: str | None = None,
    context: str | None = None,
    cluster: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = None,
    access_type: str | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: t.List[dict[str, t.Any]] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> DeploymentInfo:
    config_params = DeploymentConfigParameters(
        name=name,
        path_context=path_context,
        context=context,
        bento=bento,
        cluster=cluster,
        access_type=access_type,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=envs,
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

    return Deployment.update(
        deployment_config_params=config_params,
        context=context,
    )


@t.overload
def apply(
    name: str | None = ...,
    cluster: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    context: t.Optional[str] = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_dict: t.Optional[dict[str, t.Any]] = ...,
) -> DeploymentInfo:
    ...


@t.overload
def apply(
    name: str | None = ...,
    cluster: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    context: t.Optional[str] = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_file: t.Optional[str] = ...,
) -> DeploymentInfo:
    ...


@inject
def apply(
    name: str | None = None,
    cluster: str | None = None,
    path_context: str | None = None,
    context: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    bento: Tag | str | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> DeploymentInfo:
    config_params = DeploymentConfigParameters(
        name=name,
        path_context=path_context,
        context=context,
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
        context=context,
    )


def get(
    name: str,
    context: str | None = None,
    cluster: str | None = None,
) -> DeploymentInfo:
    return Deployment.get(
        name=name,
        context=context,
        cluster=cluster,
    )


def terminate(
    name: str,
    context: str | None = None,
    cluster: str | None = None,
) -> DeploymentInfo:
    return Deployment.terminate(
        name=name,
        context=context,
        cluster=cluster,
    )


def delete(
    name: str,
    context: str | None = None,
    cluster: str | None = None,
) -> None:
    Deployment.delete(
        name=name,
        context=context,
        cluster=cluster,
    )


def list(
    context: str | None = None,
    cluster: str | None = None,
    search: str | None = None,
) -> t.List[DeploymentInfo]:
    return Deployment.list(context=context, cluster=cluster, search=search)


__all__ = ["create", "get", "update", "apply", "terminate", "delete", "list"]
