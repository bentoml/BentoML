from __future__ import annotations

import os
import typing as t
import logging
from typing import TYPE_CHECKING

from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import resolve_user_filepath
from .frontend.dockerfile import DistroSpec
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


def to_bento_field(s: str):
    return f"bento__{s}"


def to_options_field(s: str):
    return f"__options__{s}"


def get_templates_variables(
    docker: DockerOptions,
    conda: CondaOptions,
    bento_fs: FS,
    *,
    _is_cuda: bool = False,
    **bento_env: str | bool,
) -> dict[str, t.Any]:
    """
    Returns a dictionary of variables to be used in BentoML base templates.
    """
    conda_python_version = conda.get_python_version(bento_fs)
    if conda_python_version is None:
        conda_python_version = docker.python_version

    if docker.base_image is not None:
        base_image = docker.base_image
        logger.info(
            "BentoML will not install Python to custom base images; ensure the base image '%s' has Python installed.",
            base_image,
        )
    else:
        spec = DistroSpec.from_options(docker, conda)
        python_version = docker.python_version
        assert docker.distro is not None and python_version is not None
        if docker.distro in ("ubi8"):
            # ubi8 base images uses "py38" instead of "py3.8" in its image tag
            python_version = python_version.replace(".", "")
        base_image = spec.image.format(spec_version=python_version)
        if docker.cuda_version is not None:
            base_image = spec.image.format(spec_version=docker.cuda_version)

    # bento__env
    default_env = {
        "uid_gid": BENTO_UID_GID,
        "user": BENTO_USER,
        "home": BENTO_HOME,
        "path": BENTO_PATH,
        "add_header": True,
        "buildkit_frontend": BENTO_BUILDKIT_FRONTEND,
        "enable_buildkit": True,
    }
    if bento_env:
        default_env.update(bento_env)

    return {
        **{to_options_field(k): v for k, v in docker.to_dict().items()},
        **{to_bento_field(k): v for k, v in default_env.items()},
        "__prometheus_port__": BentoMLContainer.grpc.metrics.port.get(),
        "__base_image__": base_image,
        "__conda_python_version__": conda_python_version,
        "__is_cuda__": _is_cuda,
    }


def generate_containerfile(
    docker: DockerOptions,
    build_ctx: str,
    *,
    conda: CondaOptions,
    bento_fs: FS,
    frontend: str = "dockerfile",
    **override_bento_env: t.Any,
) -> str:
    """
    Generate a Dockerfile that containerize a Bento.

    .. note::

        You should use ``construct_containerfile`` instead of this function.

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
    TEMPLATES_PATH = [
        os.path.join(os.path.dirname(__file__), "frontend", frontend, "templates")
    ]
    ENVIRONMENT = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(TEMPLATES_PATH, followlinks=True),
    )

    if docker.cuda_version is not None:
        release_type = "cuda"
    elif not conda.is_empty():
        release_type = "miniconda"
    else:
        release_type = "python"
    base = f"{release_type}_{docker.distro}.j2"
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
        dir_path = os.path.dirname(resolve_user_filepath(user_templates, build_ctx))
        user_templates = os.path.basename(user_templates)
        TEMPLATES_PATH.append(dir_path)
        environment = ENVIRONMENT.overlay(
            loader=FileSystemLoader(TEMPLATES_PATH, followlinks=True)
        )
        template = environment.get_template(
            user_templates,
            globals={"bento_base_template": template, **J2_FUNCTION},
        )

    return template.render(
        **get_templates_variables(
            docker,
            conda,
            bento_fs,
            _is_cuda=release_type == "cuda",
            **override_bento_env,
        )
    )
