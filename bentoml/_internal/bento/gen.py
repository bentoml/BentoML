"""
templates generation for all bento dockerfiles, using jinja2 as backend.

All given `__<fields>__` shouldn't be accessed by external users.

All exported blocks that users can use to extends have format: `SETUP_BENTOML_<functionality>`
  - SETUP_BENTO_BASE_IMAGE
  - SETUP_BENTO_USER
  - SETUP_BENTO_ENVARS
  - SETUP_BENTO_COMPONENTS
  - SETUP_BENTO_ENTRYPOINT
Users are free to create/add their own block. However, when using predefined bento blocks, only the aboved blocks are allowed.

All given `bento__<fields>` users can access and use to modify. Each of these will always have a default fallback.
"""
from __future__ import annotations

import os
import re
import typing as t
import logging
from typing import TYPE_CHECKING

import attr
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import bentoml_cattr
from .docker import DistroSpec
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from .build_dev_bentoml_whl import BENTOML_DEV_BUILD

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
    return os.path.expandvars(os.path.join(BENTO_PATH, *path))


BENTO_UID_GID = 1034
BENTO_USER = "bentoml"
BENTO_HOME = f"/home/{BENTO_USER}/"
BENTO_PATH = f"{BENTO_HOME}bento"


J2_FUNCTION: dict[str, GenericFunc[t.Any]] = {
    "clean_bentoml_version": clean_bentoml_version,
    "expands_bento_path": expands_bento_path,
}


@attr.frozen(on_setattr=None, eq=False, repr=False)
class ReservedEnv:
    base_image: str
    supported_architectures: t.List[str]
    bentoml_version: str = attr.field(default=BENTOML_VERSION)
    is_editable: bool = attr.field(
        default=str(os.environ.get(BENTOML_DEV_BUILD, False)).lower() == "true"
    )


@attr.frozen(on_setattr=None, eq=False, repr=False)
class CustomizableEnv:
    uid_gid: int = attr.field(default=BENTO_UID_GID)
    user: str = attr.field(default=BENTO_USER)
    home: str = attr.field(default=BENTO_HOME)
    path: str = attr.field(default=BENTO_PATH)


bentoml_cattr.register_unstructure_hook(
    ReservedEnv, lambda rs: {f"__{k}__": v for k, v in attr.asdict(rs).items()}
)
attr.resolve_types(ReservedEnv, globals(), locals())

bentoml_cattr.register_unstructure_hook(
    CustomizableEnv, lambda rs: {f"bento__{k}": v for k, v in attr.asdict(rs).items()}
)

attr.resolve_types(CustomizableEnv, globals(), locals())


def get_template_env(
    docker_options: DockerOptions, spec: DistroSpec
) -> dict[str, t.Any]:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version
    python_version = docker_options.python_version

    supported_architectures = spec.supported_architectures

    if docker_options.base_image is None:
        if cuda_version is not None:
            base_image = spec.image.format(spec_version=cuda_version)
        else:
            if distro in ["ubi8"]:
                python_version = python_version.replace(".", "")
            else:
                python_version = python_version
            base_image = spec.image.format(spec_version=python_version)
    else:
        base_image = docker_options.base_image
        logger.info(
            f"BentoML will not install Python to custom base images; ensure the base image '{base_image}' has Python installed."
        )

    return {
        **{
            f"__options__{k}": v
            for k, v in bentoml_cattr.unstructure(docker_options).items()
        },
        **bentoml_cattr.unstructure(CustomizableEnv()),
        **bentoml_cattr.unstructure(ReservedEnv(base_image, supported_architectures)),
    }


def validate_setup_blocks(environment: Environment, dockerfile_template: str) -> None:
    """
    Validate all setup blocks in the given environment.
    """
    base_blocks = set(environment.get_template("base.j2").blocks)
    user_blocks = set(environment.get_template(dockerfile_template).blocks)

    contains_bentoml_blocks = set(filter(lambda x: "SETUP_BENTO" in x, user_blocks))
    if not contains_bentoml_blocks.issubset(base_blocks):
        raise BentoMLException(
            f"Unknown SETUP block in `{dockerfile_template}`: {list(filter(lambda x: x not in base_blocks, contains_bentoml_blocks))}. All supported blocks include: {', '.join(base_blocks)}"
        )


def generate_dockerfile(docker_options: DockerOptions, *, use_conda: bool) -> str:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version
    user_templates = docker_options.dockerfile_template

    templates_path = [os.path.join(os.path.dirname(__file__), "docker", "templates")]
    if user_templates is not None:
        dir_path = os.path.dirname(os.path.realpath(user_templates))
        templates_path.append(dir_path)

    spec = DistroSpec.from_distro(
        distro, cuda=cuda_version is not None, conda=use_conda
    )
    if spec is None:
        raise BentoMLException("function is called before with_defaults() is invoked.")

    j2_template = f"{spec.release_type}_{distro}.j2"

    dockerfile_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(templates_path, followlinks=True),
    )
    dockerfile_env.globals.update(**J2_FUNCTION)  # type: ignore

    validate_setup_blocks(dockerfile_env, j2_template)
    bento_dockerfile_tmpl = dockerfile_env.get_template(j2_template)

    if user_templates is not None:
        template = dockerfile_env.get_template(
            user_templates,
            globals={"bento__dockerfile": bento_dockerfile_tmpl},
        )
        validate_setup_blocks(dockerfile_env, user_templates)
    else:
        template = bento_dockerfile_tmpl

    return template.render(**get_template_env(docker_options, spec))
