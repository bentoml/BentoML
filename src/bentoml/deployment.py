"""
User facing python APIs for deployment
"""

from __future__ import annotations

import typing as t

from simple_di import Provide
from simple_di import inject

from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.tag import Tag

from ._internal.configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from ._internal.bento import BentoStore
    from ._internal.cloud import BentoCloudClient


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = ...,
    cluster_name: str | None = ...,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: Tag | str | None = ...,
    cluster_name: str | None = ...,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: Tag | str | None = ...,
    config_file: str | None = ...,
) -> Deployment:
    ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = ...,
    config_file: str | None = ...,
) -> Deployment:
    ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: Tag | str | None = ...,
    config_dct: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = ...,
    config_dct: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@inject
def create(
    name: str | None = None,
    path_context: str | None = None,
    context: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = None,
    bento: Tag | str | None = None,
    cluster_name: str | None = None,
    access_type: str | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: t.List[dict[str, t.Any]] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dct: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> Deployment:
    return Deployment.create(
        project_path=project_path,
        bento=bento,
        access_type=access_type,
        name=name,
        cluster_name=cluster_name,
        scaling_min=scaling_min,
        scaling_max=scaling_max,
        instance_type=instance_type,
        strategy=strategy,
        envs=envs,
        extras=extras,
        config_dct=config_dct,
        config_file=config_file,
        path_context=path_context,
        context=context,
        _bento_store=_bento_store,
        _cloud_client=_cloud_client,
    )


@t.overload
def update(
    name: str,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster_name: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = ...,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@t.overload
def update(
    name: str,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster_name: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: Tag | str | None = ...,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@t.overload
def update(
    name: str,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster_name: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = ...,
    config_file: str | None = ...,
) -> Deployment:
    ...


@t.overload
def update(
    name: str,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster_name: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: Tag | str | None = ...,
    config_file: str | None = ...,
) -> Deployment:
    ...


@t.overload
def update(
    name: str,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster_name: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = ...,
    config_dct: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@t.overload
def update(
    name: str,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster_name: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: Tag | str | None = ...,
    config_dct: dict[str, t.Any] | None = ...,
) -> Deployment:
    ...


@inject
def update(
    name: str,
    path_context: str | None = None,
    context: str | None = None,
    cluster_name: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
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
    config_dct: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> Deployment:
    return Deployment.update(
        project_path=project_path,
        bento=bento,
        access_type=access_type,
        name=name,
        cluster_name=cluster_name,
        scaling_min=scaling_min,
        scaling_max=scaling_max,
        instance_type=instance_type,
        strategy=strategy,
        envs=envs,
        extras=extras,
        config_dct=config_dct,
        config_file=config_file,
        path_context=path_context,
        context=context,
        _bento_store=_bento_store,
        _cloud_client=_cloud_client,
    )


@t.overload
def apply(
    name: str,
    cluster_name: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    context: t.Optional[str] = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: t.Optional[str] = ...,
    config_dct: t.Optional[dict[str, t.Any]] = ...,
) -> Deployment:
    ...


@t.overload
def apply(
    name: str,
    cluster_name: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    context: t.Optional[str] = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_dct: t.Optional[dict[str, t.Any]] = ...,
) -> Deployment:
    ...


@t.overload
def apply(
    name: str,
    cluster_name: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    context: t.Optional[str] = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: t.Optional[str] = ...,
    config_file: t.Optional[str] = ...,
) -> Deployment:
    ...


@t.overload
def apply(
    name: str,
    cluster_name: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    context: t.Optional[str] = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_file: t.Optional[str] = ...,
) -> Deployment:
    ...


@inject
def apply(
    name: str,
    cluster_name: str | None = None,
    path_context: str | None = None,
    context: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    *,
    project_path: str | None = None,
    bento: Tag | str | None = None,
    config_dct: dict[str, t.Any] | None = None,
    config_file: str | None = None,
) -> Deployment:
    return Deployment.apply(
        name=name,
        project_path=project_path,
        bento=bento,
        cluster_name=cluster_name,
        context=context,
        path_context=path_context,
        _bento_store=_bento_store,
        _cloud_client=_cloud_client,
        config_dct=config_dct,
        config_file=config_file,
    )


def get(
    name: str,
    context: str | None = None,
    cluster_name: str | None = None,
    kube_namespace: str | None = None,
) -> Deployment:
    return Deployment.get(
        name=name,
        context=context,
        cluster_name=cluster_name,
        kube_namespace=kube_namespace,
    )


def terminate(
    name: str,
    context: str | None = None,
    cluster_name: str | None = None,
    kube_namespace: str | None = None,
) -> Deployment:
    return Deployment.terminate(
        name=name,
        context=context,
        cluster_name=cluster_name,
        kube_namespace=kube_namespace,
    )


def delete(
    name: str,
    context: str | None = None,
    cluster_name: str | None = None,
    kube_namespace: str | None = None,
) -> None:
    Deployment.delete(
        name=name,
        context=context,
        cluster_name=cluster_name,
        kube_namespace=kube_namespace,
    )


def list(
    context: str | None = None,
    cluster_name: str | None = None,
    search: str | None = None,
) -> t.List[Deployment]:
    return Deployment.list(context=context, cluster_name=cluster_name, search=search)


__all__ = ["create", "get", "update", "apply", "terminate", "delete", "list"]
