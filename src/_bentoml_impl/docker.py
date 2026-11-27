from __future__ import annotations

import typing as t
from pathlib import Path

import attrs

from bentoml._internal.bento.bento import ImageInfo
from bentoml._internal.container.generate import DEFAULT_BENTO_ENVS
from bentoml._internal.container.generate import to_bento_field
from bentoml._internal.container.generate import to_options_field


def get_templates_variables(
    image: ImageInfo, bento_fs: Path, **bento_envs: t.Any
) -> dict[str, t.Any]:
    from bentoml._internal.configuration.containers import BentoMLContainer

    bento_envs = {**DEFAULT_BENTO_ENVS, **bento_envs}
    options = attrs.asdict(image)
    return {
        **{to_options_field(k): v for k, v in options.items()},
        **{to_bento_field(k): v for k, v in bento_envs.items()},
        "__prometheus_port__": BentoMLContainer.grpc.metrics.port.get(),
    }


def generate_dockerfile(
    image: ImageInfo,
    bento_fs: Path,
    *,
    frontend: str = "dockerfile",
    **bento_envs: t.Any,
) -> str:
    from bentoml._internal.container.generate import build_environment

    environment = build_environment()
    template = environment.get_template("base_v2.j2")
    return template.render(**get_templates_variables(image, bento_fs, **bento_envs))
