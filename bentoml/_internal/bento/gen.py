"""
templates generation for all bento dockerfiles, using jinja2 as backend.

All given `__<fields>__` shouldn't be accessed by external users.

All exported blocks that users can use to extends have format: `SETUP_BENTOML_<functionality>` 
  - SETUP_BENTO_BASE_IMAGE
  - SETUP_BENTO_USER 
  - SETUP_BENTO_ENVARS
  - SETUP_BENTO_COMPONENTS
  - SETUP_BENTO_ENTRYPOINT

All given `bento__<fields>` users can access and use to modify. Each of these will always have a default fallback.
"""
from __future__ import annotations

import re
import typing as t
import logging
from typing import TYPE_CHECKING
import os

import fs
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from .build_dev_bentoml_whl import BENTOML_DEV_BUILD

from ..utils import bentoml_cattr
from .docker import DistroSpec
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from .build_config import DockerOptions

    TemplateFunc = t.Callable[[DockerOptions], t.Dict[str, t.Any]]
    GenericFunc = t.Callable[P, t.Any]


def clean_bentoml_version(bentoml_version: str) -> str:
    post_version = bentoml_version.split("+")[0]
    match = re.match(r"^(\d+).(\d+).(\d+)(?:(a|rc)\d)", post_version)
    if match is None:
        raise BentoMLException("Errors while parsing BentoML version.")
    return match.group()


def expands_bento_path(*path: str):
    return os.path.expandvars(fs.path.join(BENTO_DEFAULT_PATH, *path))


BENTO_DEFAULT_UID_GID = 1034
BENTO_DEFAULT_USER = "bentoml"
BENTO_DEFAULT_HOME = f"/home/{BENTO_DEFAULT_USER}"
BENTO_DEFAULT_PATH = f"{BENTO_DEFAULT_HOME}/bento"


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

    # users shouldn't touch this
    reserved_env: dict[str, t.Any] = {
        "base_image": base_image,
        "bentoml_version": BENTOML_VERSION,
        "is_editable": os.environ.get(BENTOML_DEV_BUILD, False).lower() == "true",  # type: ignore
        "supported_architecture": distro_spec.supported_architecture,
    }

    customizable_env = {
        "default_uid_gid": BENTO_DEFAULT_UID_GID,
        "default_user": BENTO_DEFAULT_USER,
        "default_path": BENTO_DEFAULT_PATH,
        "default_home": BENTO_DEFAULT_HOME,
    }

    return {
        **{f"__{k}__": v for k, v in reserved_env.items()},
        **{
            f"__options_{k}": v for k, v in bentoml_cattr.unstructure(docker_options).items()  # type: ignore
        },
        **{f"bento__{k}": v for k, v in customizable_env.items()},
    }


J2_FUNCTION: dict[str, GenericFunc[t.Any]] = {
    "clean_bentoml_version": clean_bentoml_version,
    "expands_bento_path": expands_bento_path,
}

def validate_setup_blocks(environment: Environment, dockerfile_template: str) -> None:
    """
    Validate all setup blocks in the given environment.
    """
    base_template = environment.get_template("base.j2")
    user_template = environment.get_template(dockerfile_template)
    if not set(user_template.blocks).issubset(set(base_template.blocks)):
        raise BentoMLException(
            f"Missing setup blocks in {dockerfile_template} template: {set(user_template.blocks) - set(base_template.blocks)}"
        )


def generate_dockerfile(docker_options: DockerOptions) -> str:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version
    templates_path = [fs.path.join(fs.path.dirname(__file__), "docker", "templates")]
    if docker_options.dockerfile_template:
        dir_path = os.path.dirname(os.path.realpath(docker_options.dockerfile_template))
        templates_path.append(dir_path)

    distro_spec = DistroSpec.from_distro(distro, cuda=cuda_version not in ("", None))

    j2_template = f"{distro_spec.release_type}_{distro.split('-')[0]}.j2"
    dockerfile_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(templates_path, followlinks=True),
    )
    dockerfile_env.globals.update(**J2_FUNCTION)  # type: ignore

    bento_dockerfile_tmpl = dockerfile_env.get_template(j2_template)

    if docker_options.dockerfile_template:
        template = dockerfile_env.get_template(
            docker_options.dockerfile_template,
            globals={"bento__dockerfile": bento_dockerfile_tmpl},
        )
    else:
        template = bento_dockerfile_tmpl
    print(template.blocks)

    return template.render(**get_template_env(docker_options))
