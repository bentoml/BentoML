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
    project_path: str | None = ...,
    bento: Tag | str | None = ...,
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    *,
    cluster: str | None = ...,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    ...


@t.overload
def create(
    project_path: str | None = ...,
    bento: Tag | str | None = ...,
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    *,
    config_file: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    ...


@t.overload
def create(
    project_path: str | None = ...,
    bento: Tag | str | None = ...,
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    *,
    config_dct: dict[str, t.Any] | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    ...


@inject
def create(
    project_path: str | None = None,
    bento: Tag | str | None = None,
    name: str | None = None,
    path_context: str | None = None,
    context: str | None = None,
    *,
    cluster: str | None = None,
    access_type: str | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: t.List[dict[str, t.Any]] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dct: dict[str, t.Any] | None = None,
    config_file: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    return Deployment.create_deployment(
        project_path=project_path,
        bento=bento,
        access_type=access_type,
        name=name,
        cluster=cluster,
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
    project_path: str | None = ...,
    bento: Tag | str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = ...,
    *,
    access_type: str | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[dict[str, t.Any]] | None = ...,
    extras: dict[str, t.Any] | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    ...


@t.overload
def update(
    name: str,
    project_path: str | None = ...,
    bento: Tag | str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = None,
    *,
    config_file: str | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    ...


@t.overload
def update(
    name: str,
    project_path: str | None = ...,
    bento: Tag | str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = None,
    *,
    config_dct: dict[str, t.Any] | None = ...,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    ...


@inject
def update(
    name: str,
    project_path: str | None = None,
    bento: Tag | str | None = None,
    path_context: str | None = None,
    context: str | None = None,
    cluster: str | None = None,
    *,
    access_type: str | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: t.List[dict[str, t.Any]] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dct: dict[str, t.Any] | None = None,
    config_file: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    return Deployment.update_deployment(
        project_path=project_path,
        bento=bento,
        access_type=access_type,
        name=name,
        cluster=cluster,
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


def get(
    deployment_name: str,
    context: str | None = None,
    cluster_name: str | None = None,
    kube_namespace: str | None = None,
) -> Deployment:
    return Deployment.get_deployment(
        deployment_name=deployment_name,
        context=context,
        cluster_name=cluster_name,
        kube_namespace=kube_namespace,
    )


def terminate(
    deployment_name: str,
    context: str | None = None,
    cluster_name: str | None = None,
    kube_namespace: str | None = None,
) -> Deployment:
    return Deployment.terminate_deployment(
        deployment_name=deployment_name,
        context=context,
        cluster_name=cluster_name,
        kube_namespace=kube_namespace,
    )


def delete(
    deployment_name: str,
    context: str | None = None,
    cluster_name: str | None = None,
    kube_namespace: str | None = None,
) -> Deployment:
    return Deployment.delete_deployment(
        deployment_name=deployment_name,
        context=context,
        cluster_name=cluster_name,
        kube_namespace=kube_namespace,
    )


__all__ = ["create", "get", "update", "terminate", "delete"]
