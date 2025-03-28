"""
User facing python APIs for deployment
"""

from __future__ import annotations

import typing as t

import attr
from simple_di import Provide
from simple_di import inject

from ._internal.cloud.deployment import Deployment
from ._internal.cloud.deployment import DeploymentConfigParameters
from ._internal.cloud.schemas.modelschemas import EnvItemSchema
from ._internal.cloud.schemas.modelschemas import LabelItemSchema
from ._internal.configuration.containers import BentoMLContainer
from ._internal.tag import Tag
from .exceptions import BentoMLException

if t.TYPE_CHECKING:
    from ._internal.cloud import BentoCloudClient


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
) -> Deployment: ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    *,
    bento: Tag | str | None = ...,
    config_file: str | None = ...,
) -> Deployment: ...


@t.overload
def create(
    name: str | None = ...,
    path_context: str | None = ...,
    *,
    bento: Tag | str | None = ...,
    config_dict: dict[str, t.Any] | None = ...,
) -> Deployment: ...


@inject
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
    args: dict[str, t.Any] | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    from ._internal.utils.args import set_arguments

    if args is not None:
        set_arguments(**args)
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
    return _cloud_client.deployment.create(deployment_config_params=config_params)


@t.overload
def update(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = ...,
    *,
    bento: Tag | str | None = ...,
    access_authorization: bool | None = ...,
    scaling_min: int | None = ...,
    scaling_max: int | None = ...,
    instance_type: str | None = ...,
    strategy: str | None = ...,
    envs: t.List[EnvItemSchema] | t.List[dict[str, t.Any]] | None = ...,
    secrets: t.List[str] | None = ...,
    extras: dict[str, t.Any] | None = ...,
) -> Deployment: ...


@t.overload
def update(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = None,
    *,
    bento: Tag | str | None = ...,
    config_file: str | None = ...,
) -> Deployment: ...


@t.overload
def update(
    name: str | None = ...,
    path_context: str | None = ...,
    context: str | None = ...,
    cluster: str | None = None,
    *,
    bento: Tag | str | None = ...,
    config_dict: dict[str, t.Any] | None = ...,
) -> Deployment: ...


@inject
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
    secrets: t.List[str] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
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
        secrets=secrets,
        extras=extras,
        config_dict=config_dict,
        config_file=config_file,
    )
    try:
        config_params.verify(create=False)
    except BentoMLException as e:
        raise BentoMLException(
            f"Failed to create deployment due to invalid configuration: {e}"
        )

    return _cloud_client.deployment.update(deployment_config_params=config_params)


@t.overload
def apply(
    name: str | None = ...,
    cluster: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_dict: t.Optional[dict[str, t.Any]] = ...,
) -> Deployment: ...


@t.overload
def apply(
    name: str | None = ...,
    cluster: t.Optional[str] = ...,
    path_context: t.Optional[str] = ...,
    *,
    bento: t.Optional[t.Union[Tag, str]] = ...,
    config_file: t.Optional[str] = ...,
) -> Deployment: ...


@inject
def apply(
    name: str | None = None,
    cluster: str | None = None,
    path_context: str | None = None,
    *,
    bento: Tag | str | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    config_params = DeploymentConfigParameters(
        name=name,
        path_context=path_context,
        bento=bento,
        cluster=cluster,
        config_dict=config_dict,
        config_file=config_file,
    )
    try:
        config_params.verify(create=False)
    except BentoMLException as e:
        raise BentoMLException(
            f"Failed to create deployment due to invalid configuration: {e}"
        )

    return _cloud_client.deployment.apply(deployment_config_params=config_params)


@inject
def get(
    name: str,
    cluster: str | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    return _cloud_client.deployment.get(name=name, cluster=cluster)


@inject
def terminate(
    name: str,
    cluster: str | None = None,
    wait: bool = False,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    return _cloud_client.deployment.terminate(name=name, cluster=cluster, wait=wait)


@inject
def delete(
    name: str,
    cluster: str | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    _cloud_client.deployment.delete(name=name, cluster=cluster)


@inject
def list(
    cluster: str | None = None,
    search: str | None = None,
    dev: bool = False,
    q: str | None = None,
    labels: t.List[LabelItemSchema] | t.List[dict[str, t.Any]] | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> t.List[Deployment]:
    # Syntatic sugar to enable searching by `labels` argument
    if labels is not None:
        label_query = " ".join(
            f"label:{d['key']}={d['value']}"
            for d in (
                label if isinstance(label, dict) else attr.asdict(label)
                for label in labels
            )
        )

        if q is not None:
            q = f"{q} {label_query}"
        else:
            q = label_query

    return _cloud_client.deployment.list(cluster=cluster, search=search, dev=dev, q=q)


__all__ = ["create", "get", "update", "apply", "terminate", "delete", "list"]
