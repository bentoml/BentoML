"""
User facing python APIs for deployment
"""

from __future__ import annotations

import typing as t

from simple_di import Provide
from simple_di import inject

from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.deployment import DeploymentInfo
from bentoml._internal.cloud.deployment import get_args_from_config
from bentoml._internal.cloud.deployment import get_real_bento_tag
from bentoml._internal.tag import Tag
from bentoml.cloud import BentoCloudClient
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
    project_path: str | None = ...,
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
    project_path: str | None = ...,
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


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    *,
    project_path: str | None = ...,
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
    project_path: str | None = None,
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
    deploy_by_param = (
        access_type
        or cluster
        or scaling_min
        or scaling_max
        or instance_type
        or strategy
        or envs
        or extras
    )
    if (
        config_dict
        and config_file
        or config_dict
        and deploy_by_param
        or config_file
        and deploy_by_param
    ):
        raise BentoMLException(
            "Configure a deployment can only use one of the following: config_dict, config_file, or the other parameters"
        )
    deploy_name, bento_name, cluster_name = get_args_from_config(
        bento=bento,
        name=name,
        cluster=cluster,
        config_dict=config_dict,
        config_file=config_file,
        path_context=path_context,
    )

    if bento_name and project_path:
        raise BentoMLException("Only one of bento or project_path can be provided")
    if bento_name is None and project_path is None:
        raise BentoMLException("Either bento or project_path must be provided")
    bento = get_real_bento_tag(
        project_path=project_path,
        bento=bento_name,
        context=context,
        _cloud_client=BentoCloudClient(),
    )

    return Deployment.create(
        bento=bento,
        access_type=access_type,
        name=deploy_name,
        cluster=cluster_name,
        scaling_min=scaling_min,
        scaling_max=scaling_max,
        instance_type=instance_type,
        strategy=strategy,
        envs=envs,
        extras=extras,
        config_dict=config_dict,
        config_file=config_file,
        path_context=path_context,
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
    project_path: str | None = ...,
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
    project_path: str | None = ...,
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
    project_path: str | None = ...,
    config_dict: dict[str, t.Any] | None = ...,
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
    project_path: str | None = None,
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
    deploy_by_param = (
        access_type
        or scaling_min
        or scaling_max
        or instance_type
        or strategy
        or envs
        or extras
    )
    if (
        config_dict
        and config_file
        or config_dict
        and deploy_by_param
        or config_file
        and deploy_by_param
    ):
        raise BentoMLException(
            "Configure a deployment can only use one of the following: config_dict, config_file, or the other parameters"
        )
    deploy_name, bento_name, cluster_name = get_args_from_config(
        bento=bento,
        name=name,
        cluster=cluster,
        config_dict=config_dict,
        config_file=config_file,
        path_context=path_context,
    )
    if bento_name and project_path:
        raise BentoMLException("Only one of bento or project_path can be provided")
    if bento_name is None and project_path is None:
        bento = None
    else:
        bento = get_real_bento_tag(
            project_path=project_path,
            bento=bento_name,
            context=context,
            _cloud_client=BentoCloudClient(),
        )

    return Deployment.update(
        bento=bento,
        access_type=access_type,
        name=deploy_name,
        cluster=cluster_name,
        scaling_min=scaling_min,
        scaling_max=scaling_max,
        instance_type=instance_type,
        strategy=strategy,
        envs=envs,
        extras=extras,
        config_dict=config_dict,
        config_file=config_file,
        path_context=path_context,
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
    project_path: t.Optional[str] = ...,
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
    project_path: t.Optional[str] = ...,
    config_file: t.Optional[str] = ...,
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
    project_path: str | None = None,
    bento: Tag | str | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> DeploymentInfo:
    if config_dict and config_file:
        raise BentoMLException(
            "Configure a deployment can only use one of the following: config_dict, config_file"
        )
    deploy_name, bento_name, cluster_name = get_args_from_config(
        bento=bento,
        name=name,
        cluster=cluster,
        config_dict=config_dict,
        config_file=config_file,
        path_context=path_context,
    )
    if bento_name and project_path:
        raise BentoMLException("Only one of bento or project_path can be provided")
    if bento_name is None and project_path is None:
        bento = None
    else:
        bento = get_real_bento_tag(
            project_path=project_path,
            bento=bento_name,
            context=context,
            _cloud_client=BentoCloudClient(),
        )

    return Deployment.apply(
        name=deploy_name,
        bento=bento,
        cluster=cluster_name,
        context=context,
        path_context=path_context,
        config_dict=config_dict,
        config_file=config_file,
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
