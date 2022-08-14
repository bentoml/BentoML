from __future__ import annotations

import os
import typing as t
import logging
from sys import version_info
from typing import TYPE_CHECKING
from dataclasses import asdict
from dataclasses import dataclass

from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from .docker import DistroSpec
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    from .build_config import DockerOptions

    TemplateFunc = t.Callable[[DockerOptions], t.Dict[str, t.Any]]
    F = t.Callable[P, t.Any]

BENTO_UID_GID = 1034
BENTO_USER = "bentoml"
BENTO_HOME = f"/home/{BENTO_USER}/"
BENTO_PATH = f"{BENTO_HOME}bento"
BLOCKS = {
    "SETUP_BENTO_BASE_IMAGE",
    "SETUP_BENTO_USER",
    "SETUP_BENTO_ENVARS",
    "SETUP_BENTO_COMPONENTS",
    "SETUP_BENTO_ENTRYPOINT",
}


def expands_bento_path(*path: str, bento_path: str = BENTO_PATH) -> str:
    """
    Expand a given paths with respect to :code:`BENTO_PATH`.
    Note on using "/": the returned path is meant to be used in the generated Dockerfile.
    """
    return "/".join([bento_path, *path])


J2_FUNCTION: dict[str, F[t.Any]] = {"expands_bento_path": expands_bento_path}

to_preserved_field: t.Callable[[str], str] = lambda s: f"__{s}__"
to_bento_field: t.Callable[[str], str] = lambda s: f"bento__{s}"
to_options_field: t.Callable[[str], str] = lambda s: f"__options__{s}"


@dataclass
class ReservedEnv:
    base_image: str
    python_version: str = f"{version_info.major}.{version_info.minor}"

    def asdict(self) -> dict[str, t.Any]:
        return {
            **{to_preserved_field(k): v for k, v in asdict(self).items()},
            "__prometheus_port__": BentoMLContainer.grpc.metrics.port.get(),
        }


@dataclass
class CustomizableEnv:
    uid_gid: int = BENTO_UID_GID
    user: str = BENTO_USER
    home: str = BENTO_HOME
    path: str = BENTO_PATH

    def asdict(self) -> dict[str, t.Any]:
        return {to_bento_field(k): v for k, v in asdict(self).items()}


def get_templates_variables(
    options: DockerOptions, use_conda: bool
) -> dict[str, t.Any]:
    """
    Returns a dictionary of variables to be used in BentoML base templates.
    """

    if options.base_image is None:
        distro = options.distro
        cuda_version = options.cuda_version
        python_version = options.python_version

        # these values will be set at with_defaults() if not provided
        # so distro and python_version won't be None here.
        assert distro and python_version
        spec = DistroSpec.from_distro(
            distro, cuda=cuda_version is not None, conda=use_conda
        )
        if cuda_version is not None:
            base_image = spec.image.format(spec_version=cuda_version)
        else:
            if distro in ["ubi8"]:
                # ubi8 base images uses "py38" instead of "py3.8" in its image tag
                python_version = python_version.replace(".", "")
            else:
                python_version = python_version
            base_image = spec.image.format(spec_version=python_version)
    else:
        base_image = options.base_image
        logger.info(
            "BentoML will not install Python to custom base images; ensure the base image '%s' has Python installed.",
            base_image,
        )

    # environment returns are
    # __base_image__, __python_version__, __prometheus_port__
    # bento__uid_gid, bento__user, bento__home, bento__path
    # __options__distros, __options__base_image, __options_env, __options_system_packages, __options_setup_script
    return {
        **{
            to_options_field(k): v
            for k, v in bentoml_cattr.unstructure(options).items()
        },
        **CustomizableEnv().asdict(),
        **ReservedEnv(base_image=base_image).asdict(),
    }


def generate_dockerfile(
    options: DockerOptions,
    build_ctx: str,
    *,
    use_conda: bool,
) -> str:
    """
    Generate a Dockerfile that containerize a Bento.

    Args:
        docker_options (:class:`DockerOptions`):
            Docker Options class parsed from :code:`bentofile.yaml` or any arbitrary Docker options class.
        use_conda (:code:`bool`, `required`):
            Whether to use conda in the given Dockerfile.

    Returns:
        str: The rendered Dockerfile string.

    .. dropdown:: Implementation Notes

        The coresponding Dockerfile template will be determined automatically based on given :class:`DockerOptions`.
        The templates are located `here <https://github.com/bentoml/BentoML/tree/main/bentoml/_internal/bento/docker/templates>`_

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

        Each of the templates will be validated correctly before being rendered. This will also include the user-defined custom templates
        if :code:`dockerfile_template` is specified.

    .. code-block:: python

        from bentoml._internal.bento.gen import generate_dockerfile
        from bentoml._internal.bento.bento import BentoInfo

        docker_options = BentoInfo.from_yaml_file("{bento_path}/bento.yaml").docker
        docker_options.dockerfile_template = "./override_template.j2"
        dockerfile = generate_dockerfile(docker_options, use_conda=False)

    """
    distro = options.distro
    use_cuda = options.cuda_version is not None
    user_templates = options.dockerfile_template

    TEMPLATES_PATH = [os.path.join(os.path.dirname(__file__), "docker", "templates")]
    ENVIRONMENT = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(TEMPLATES_PATH, followlinks=True),
    )

    if options.base_image is not None:
        base = "base.j2"
    else:
        assert distro  # distro will be set via 'with_defaults()'
        spec = DistroSpec.from_distro(distro, cuda=use_cuda, conda=use_conda)
        base = f"{spec.release_type}_{distro}.j2"

    template = ENVIRONMENT.get_template(base, globals=J2_FUNCTION)
    logger.debug(
        f"Using base Dockerfile template: {base}, and their path: {os.path.join(TEMPLATES_PATH[0], base)}"
    )

    if user_templates is not None:
        dir_path = os.path.dirname(resolve_user_filepath(user_templates, build_ctx))
        user_templates = os.path.basename(user_templates)
        TEMPLATES_PATH.append(dir_path)
        new_loader = FileSystemLoader(TEMPLATES_PATH, followlinks=True)

        environment = ENVIRONMENT.overlay(loader=new_loader)
        template = environment.get_template(
            user_templates,
            globals={"bento_base_template": template, **J2_FUNCTION},
        )

    return template.render(**get_templates_variables(options, use_conda=use_conda))
