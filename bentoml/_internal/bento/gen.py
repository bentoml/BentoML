from __future__ import annotations

import os
import re
import typing as t
import logging
from typing import TYPE_CHECKING

import fs
import attr
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import bentoml_cattr as cattr
from ..utils import resolve_user_filepath
from .docker import DistroSpec
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from .build_dev_bentoml_whl import BENTOML_DEV_BUILD

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    from jinja2.loaders import BaseLoader
    from jinja2.environment import Template

    from .build_config import DockerOptions

    TemplateFunc = t.Callable[[DockerOptions], t.Dict[str, t.Any]]
    GenericFunc = t.Callable[P, t.Any]

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


def clean_bentoml_version(bentoml_version: str) -> str:
    post_version = bentoml_version.split("+")[0]
    match = re.match(r"^(\d+).(\d+).(\d+)(?:(a|rc)\d)*", post_version)
    if match is None:
        raise BentoMLException("Errors while parsing BentoML version.")
    return match.group()


def expands_bento_path(*path: str, bento_path: str = BENTO_PATH) -> str:
    """
    Expand a given paths with respect to :code:`BENTO_PATH`.
    """
    return os.path.expandvars(os.path.join(bento_path, *path))


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


cattr.register_unstructure_hook(
    ReservedEnv, lambda rs: {f"__{k}__": v for k, v in attr.asdict(rs).items()}
)
attr.resolve_types(ReservedEnv, globals(), locals())

cattr.register_unstructure_hook(
    CustomizableEnv, lambda rs: {f"bento__{k}": v for k, v in attr.asdict(rs).items()}
)

attr.resolve_types(CustomizableEnv, globals(), locals())

TEMPLATES_PATH = [fs.path.join(os.path.dirname(__file__), "docker", "templates")]
ENVIRONMENT = Environment(
    extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"],
    trim_blocks=True,
    lstrip_blocks=True,
    loader=FileSystemLoader(TEMPLATES_PATH, followlinks=True),
)


def validate_user_template(template: Template, loader: BaseLoader) -> None | t.NoReturn:
    """
    Validate all user-defined templates in the given environment.
    """
    ctx = template.new_context()
    if "bento__dockerfile" not in ctx.keys():
        exc_info = (
            f"User-defined template `{template}` does not contain `bento__dockerfile`. "
            + "Make sure to add {% extends bento__dockerfile %} to the top of your dockerfile template."
        )
        raise BentoMLException(exc_info)

    if template.name is None:
        raise BentoMLException(
            "Template name is invalid. Make sure to specify the correct template file under `dockerfile_template`."
        )
    source, filename, _ = loader.get_source(  # pragma: no cover (covered by jinja2)
        template.environment, template.name
    )

    # check for setup blocks
    contains_bento_blocks = set(
        filter(lambda x: x.startswith("SETUP_BENTO"), set(ctx.blocks))
    )
    if not contains_bento_blocks.issubset(BLOCKS):
        invalid_blocks = contains_bento_blocks - BLOCKS
        raise BentoMLException(
            f"Unknown SETUP block in `{filename}`: {list(invalid_blocks)}. All supported blocks include: {', '.join(BLOCKS)}"
        )

    # check for reserved env
    maybe_reserved = list(
        map(
            lambda x: x[-1],
            filter(
                lambda x: x[1] == "name" and x[-1].startswith("__"),
                template.environment.lex(source),
            ),
        )
    )
    reserved_var = [f"__{k.name}__" for k in attr.fields(ReservedEnv)]

    if set(maybe_reserved).intersection(reserved_var) or any(
        i for i in maybe_reserved if i.startswith("__options__")
    ):
        raise BentoMLException(
            "User defined Dockerfile template contains reserved variables. These variables are internally used by BentoML and should not be accessed by users. Refers to https://docs.bentoml.org/en/latest/concepts/bento.html#docker-template-danger-zone to see which variables you can use in your custom docker templates."
        )


def get_docker_variables(
    options: DockerOptions, spec: DistroSpec | None
) -> dict[str, t.Any]:
    if spec is None:
        raise BentoMLException("Distro spec is required, got None instead.")

    distro = options.distro
    cuda_version = options.cuda_version
    python_version = options.python_version

    supported_architectures = spec.supported_architectures

    if options.base_image is None:
        if cuda_version is not None:
            base_image = spec.image.format(spec_version=cuda_version)
        else:
            if distro in ["ubi8"]:
                python_version = python_version.replace(".", "")
            else:
                python_version = python_version
            base_image = spec.image.format(spec_version=python_version)
    else:
        base_image = options.base_image
        logger.info(
            f"BentoML will not install Python to custom base images; ensure the base image '{base_image}' has Python installed."
        )

    return {
        **{
            f"__options__{k}": v
            for k, v in cattr.unstructure(options).items()  # type: ignore
        },
        **cattr.unstructure(CustomizableEnv()),  # type: ignore
        **cattr.unstructure(ReservedEnv(base_image, supported_architectures)),  # type: ignore
    }


def generate_dockerfile(options: DockerOptions, *, use_conda: bool) -> str:
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

        from bentoml.build_utils import generate_dockerfile
        from bentoml.build_utils import DockerOptions

        dockerfile = generate_dockerfile(DockerOptions().with_defaults(), use_conda=False)

    """
    distro = options.distro
    use_cuda = options.cuda_version is not None
    user_templates = options.dockerfile_template

    spec = DistroSpec.from_distro(distro, cuda=use_cuda, conda=use_conda)
    if spec is None:
        raise BentoMLException("function is called before with_defaults() is invoked.")

    base = f"{spec.release_type}_{distro}.j2"
    template = ENVIRONMENT.get_template(base, globals=J2_FUNCTION)

    if user_templates is not None:
        dir_path = fs.path.dirname(resolve_user_filepath(user_templates, os.getcwd()))
        user_templates = os.path.basename(user_templates)
        TEMPLATES_PATH.append(dir_path)
        new_loader = FileSystemLoader(TEMPLATES_PATH, followlinks=True)

        environment = ENVIRONMENT.overlay(loader=new_loader)
        template = environment.get_template(
            user_templates,
            globals={"bento__dockerfile": template, **J2_FUNCTION},
        )
        validate_user_template(template, loader=new_loader)

    return template.render(**get_docker_variables(options, spec))
