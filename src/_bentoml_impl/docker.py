from __future__ import annotations

import importlib
import os
import shlex
import typing as t

import attrs
from fs.base import FS
from jinja2 import Environment
from jinja2 import FileSystemLoader

from bentoml._internal.bento.bento import ImageInfo
from bentoml._internal.container.generate import DEFAULT_BENTO_ENVS
from bentoml._internal.container.generate import PREHEAT_PIP_PACKAGES
from bentoml._internal.container.generate import expands_bento_path
from bentoml._internal.container.generate import resolve_package_versions
from bentoml._internal.container.generate import to_bento_field
from bentoml._internal.container.generate import to_options_field


def get_templates_variables(
    image: ImageInfo, bento_fs: FS, **bento_envs: t.Any
) -> dict[str, t.Any]:
    from bentoml._internal.configuration.containers import BentoMLContainer

    bento_envs = {**DEFAULT_BENTO_ENVS, **bento_envs}
    options = attrs.asdict(image)
    requirement_file = bento_fs.getsyspath("env/python/requirements.txt")
    if os.path.exists(requirement_file):
        python_packages = resolve_package_versions(requirement_file)
    else:
        python_packages = {}
    pip_preheat_packages = [
        python_packages[k] for k in PREHEAT_PIP_PACKAGES if k in python_packages
    ]
    return {
        **{to_options_field(k): v for k, v in options.items()},
        **{to_bento_field(k): v for k, v in bento_envs.items()},
        "__prometheus_port__": BentoMLContainer.grpc.metrics.port.get(),
        "__pip_preheat_packages__": pip_preheat_packages,
    }


def generate_dockerfile(
    image: ImageInfo, bento_fs: FS, *, frontend: str = "dockerfile", **bento_envs: t.Any
) -> str:
    templates_path = importlib.import_module(
        f"bentoml._internal.container.frontend.{frontend}.templates"
    ).__path__
    environment = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(templates_path, followlinks=True),
    )
    environment.filters["bash_quote"] = shlex.quote
    environment.globals["expands_bento_path"] = expands_bento_path
    template = environment.get_template("base_v2.j2")
    return template.render(**get_templates_variables(image, bento_fs, **bento_envs))
