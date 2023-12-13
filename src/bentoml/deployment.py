"""
User facing python APIs for deployment
"""

from __future__ import annotations

import typing as t

from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.tag import Tag


def create(
    project_path: str | None = None,
    bento: Tag | str | None = None,
    access_type: str | None = None,
    name: str | None = None,
    cluster: str | None = None,
    scailing_min: int | None = None,
    scailing_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    envs: t.List[dict[str, t.Any]] | None = None,
    extras: dict[str, t.Any] | None = None,
    config_dct: dict[str, t.Any] | None = None,
    config_file: str | None = None,
    path_context: str | None = None,
    context: str | None = None,
):
    return Deployment.create_deploymentV2(
        project_path=project_path,
        bento=bento,
        access_type=access_type,
        name=name,
        cluster=cluster,
        scailing_min=scailing_min,
        scailing_max=scailing_max,
        instance_type=instance_type,
        strategy=strategy,
        envs=envs,
        extras=extras,
        config_dct=config_dct,
        config_file=config_file,
        path_context=path_context,
        context=context,
    )


__all__ = ["create"]
