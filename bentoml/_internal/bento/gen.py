from __future__ import annotations

import re
import typing as t
import logging
from typing import TYPE_CHECKING

import fs
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import bentoml_cattr
from .docker import DistroSpec
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from .build_config import DockerOptions

    TemplateFunc = t.Callable[[DockerOptions], t.Dict[str, t.Any]]


def clean_bentoml_version() -> str:
    post_version = BENTOML_VERSION.split("+")[0]
    match = re.match(r"^(\d+).(\d+).(\d+)(?:(a|rc)\d)", post_version)
    if match is None:
        raise BentoMLException("Errors while parsing BentoML version.")
    return match.group()


DEFAULT_UID_GID = 1034
DEFAULT_BENTO_USER = "bentoml"
DEFAULT_BENTO_HOME = f"/home/{DEFAULT_BENTO_USER}"
DEFAULT_BENTO_PATH = f"{DEFAULT_BENTO_HOME}/bento"


def get_template_env(docker_options: DockerOptions) -> dict[str, t.Any]:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version
    python_version = docker_options.python_version

    distro_spec = DistroSpec.from_distro(distro, cuda=cuda_version not in (None, ""))

    if docker_options.base_image is None:
        if cuda_version not in ("", None):
            base_image = distro_spec.image.format(cuda_version=cuda_version)
        else:
            if distro in ["ubi8"]:
                python_version = python_version.replace(".", "")
            else:
                python_version = python_version
        base_image = distro_spec.image.format(python_version=python_version)
    else:
        base_image = docker_options.base_image
        logger.warning(f"Make sure to have Python installed for {base_image}.")

    return {
        "base_image": base_image,
        "bentoml_version": clean_bentoml_version(),
        "default_uid_gid": DEFAULT_UID_GID,
        "default_bento_user": DEFAULT_BENTO_USER,
        "default_bento_path": DEFAULT_BENTO_PATH,
        "default_bento_home": DEFAULT_BENTO_HOME,
        "user_defined_image": docker_options.base_image is not None,
        "docker_options": bentoml_cattr.unstructure(docker_options),  # type: ignore
        "distro_spec": bentoml_cattr.unstructure(distro_spec),  # type: ignore
    }


def generate_dockerfile(docker_options: DockerOptions) -> str:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version

    distro_spec = DistroSpec.from_distro(distro, cuda=cuda_version not in ("", None))

    j2_template = f"{distro_spec.release_type}_{distro.split('-')[0]}.j2"
    dockerfile_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(
            fs.path.join(fs.path.dirname(__file__), "docker", "templates"),
            followlinks=True,
        ),
    )

    bento_dockerfile_tmpl = dockerfile_env.get_template(j2_template)

    if docker_options.dockerfile_template != "":
        template = dockerfile_env.get_template(
            docker_options.dockerfile_template,
            globals={"bento_dockerfile": bento_dockerfile_tmpl},
        )
    else:
        template = bento_dockerfile_tmpl

    return template.render(**get_template_env(docker_options))
