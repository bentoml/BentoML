from __future__ import annotations

import os
import typing as t
import logging
from typing import TYPE_CHECKING
from dataclasses import asdict
from dataclasses import replace
from dataclasses import dataclass

import fs
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from .dockerfile import DistroSpec
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    from fs.base import FS

    from ..bento.build_config import CondaOptions
    from ..bento.build_config import DockerOptions

    TemplateFunc = t.Callable[[DockerOptions], dict[str, t.Any]]
    F = t.Callable[P, t.Any]

BENTO_UID_GID = 1034
BENTO_USER = "bentoml"
BENTO_HOME = f"/home/{BENTO_USER}/"
BENTO_PATH = f"{BENTO_HOME}bento"

# 1.2.1 is the current docker frontend that both buildkitd and kaniko supports.
BENTO_BUILDKIT_FRONTEND = "docker/dockerfile:1.2.1"


def expands_bento_path(*path: str, bento_path: str = BENTO_PATH) -> str:
    """
    Expand a given paths with respect to :code:`BENTO_PATH`.
    Note on using "/": the returned path is meant to be used in the generated Dockerfile.
    """
    return "/".join([bento_path, *path])


J2_FUNCTION: dict[str, F[t.Any]] = {"expands_bento_path": expands_bento_path}

to_bento_field: t.Callable[[str], str] = lambda s: f"bento__{s}"
to_options_field: t.Callable[[str], str] = lambda s: f"__options__{s}"


@dataclass
class BentoEnv:
    uid_gid: int = BENTO_UID_GID
    user: str = BENTO_USER
    home: str = BENTO_HOME
    path: str = BENTO_PATH
    add_header: bool = True
    # BuildKit specifics
    buildkit_frontend: str = BENTO_BUILDKIT_FRONTEND
    enable_buildkit: bool = True

    def asdict(self) -> dict[str, t.Any]:
        return {to_bento_field(k): v for k, v in asdict(self).items()}

    def with_options(self, **kwargs: t.Any) -> BentoEnv:
        return replace(self, **kwargs)


def get_templates_variables(
    options: DockerOptions,
    spec: DistroSpec,
    conda_python_version: str,
    **bento_env: str | bool,
) -> dict[str, t.Any]:
    """
    Returns a dictionary of variables to be used in BentoML base templates.
    """

    python_version = options.python_version
    if options.distro in ("ubi8"):
        # ubi8 base images uses "py38" instead of "py3.8" in its image tag
        python_version = python_version.replace(".", "")
    base_image = spec.image.format(spec_version=python_version)
    if options.cuda_version is not None:
        base_image = spec.image.format(spec_version=options.cuda_version)
    if options.base_image is not None:
        base_image = options.base_image
        logger.info(
            "BentoML will not install Python to custom base images; ensure the base image '%s' has Python installed.",
            base_image,
        )

    return {
        **{
            to_options_field(k): v
            for k, v in bentoml_cattr.unstructure(options).items()
        },
        **BentoEnv().with_options(**bento_env).asdict(),
        "__prometheus_port__": BentoMLContainer.grpc.metrics.port.get(),
        "__base_image__": base_image,
        "__conda_python_version__": conda_python_version,
    }


def generate_dockerfile(
    docker: DockerOptions,
    build_ctx: str,
    *,
    conda: CondaOptions,
    bento_fs: FS,
    **override_bento_env: t.Any,
) -> str:
    """
    Generate a Dockerfile that containerize a Bento.

    .. note::

        You should use ``construct_dockerfile`` instead of this function.

    Returns:
        str: The rendered Dockerfile string.

    .. dropdown:: Implementation Notes

        The coresponding Dockerfile template will be determined automatically based on given :class:`DockerOptions`.
        The templates are located `here <https://github.com/bentoml/BentoML/tree/main/src/bentoml/_internal/bento/docker/templates>`_

        As one can see, the Dockerfile templates are organized with the format :code:`<release_type>_<distro>.j2` with:

        +---------------+------------------------------------------+
        | Release type  | Description                              |
        +===============+==========================================+
        | base          | A base setup for all supported distros.  |
        +---------------+------------------------------------------+
        | cuda          | CUDA-supported templates.                |
        +---------------+------------------------------------------+
        | miniconda     | Conda-supported templates.               |
        +---------------+------------------------------------------+
        | python        | Python releases.                         |
        +---------------+------------------------------------------+

        All templates will have the following blocks: "SETUP_BENTO_BASE_IMAGE", "SETUP_BENTO_USER", "SETUP_BENTO_ENVARS", "SETUP_BENTO_COMPONENTS", "SETUP_BENTO_ENTRYPOINT",

        Overriding templates variables: bento__uid_gid, bento__user, bento__home, bento__path, bento__enable_buildkit
    """
    from ..bento.build_config import CONDA_ENV_YAML_FILE_NAME

    distro = docker.distro
    environment_yml = bento_fs.getsyspath(
        fs.path.join(
            "env",
            "conda",
            CONDA_ENV_YAML_FILE_NAME,
        )
    )
    conda_python_version = conda.get_python_version(environment_yml)
    if conda_python_version is None:
        logger.debug(
            "No python version is specified under '%s'. Using the current Python%s",
            environment_yml,
            docker.python_version,
        )
        conda_python_version = docker.python_version

    TEMPLATES_PATH = [
        os.path.join(os.path.dirname(__file__), "dockerfile", "templates")
    ]
    ENVIRONMENT = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(TEMPLATES_PATH, followlinks=True),
    )

    spec = DistroSpec.from_options(docker, conda)
    base = f"{spec.release_type}_{distro}.j2"
    if docker.base_image is not None:
        # If base_image is specified, then use the base template instead.
        base = "base.j2"

    template = ENVIRONMENT.get_template(base, globals=J2_FUNCTION)
    logger.debug(
        'Using base Dockerfile template: "%s" (path: "%s")',
        base,
        os.path.join(TEMPLATES_PATH[0], base),
    )

    user_templates = docker.dockerfile_template
    if user_templates is not None:
        TEMPLATES_PATH.append(
            os.path.dirname(resolve_user_filepath(user_templates, build_ctx))
        )
        environment = ENVIRONMENT.overlay(
            loader=FileSystemLoader(TEMPLATES_PATH, followlinks=True)
        )
        template = environment.get_template(
            os.path.basename(user_templates),
            globals={"bento_base_template": template, **J2_FUNCTION},
        )

    return template.render(
        **get_templates_variables(
            docker,
            spec=spec,
            conda_python_version=conda_python_version,
            **override_bento_env,
        )
    )
