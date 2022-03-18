from __future__ import annotations

import io
import typing as t
import logging
from typing import TYPE_CHECKING
from functools import wraps
from importlib.metadata import version

import attrs
import click
import attrs.converters
from click import ClickException
from fs.base import FS
from simple_di import inject
from simple_di import Provide
from rich.console import Console

from ._make import BuildCtx
from ._make import ReleaseCtx
from ._make import set_generation_context
from ._funcs import raise_exception
from .exceptions import ManagerException
from ._configuration import get_manifest_info
from ._configuration import SUPPORTED_OS_RELEASES
from ._configuration import DockerManagerContainer
from ._configuration import SUPPORTED_PYTHON_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")

    GenericDict = t.Dict[str, t.Any]

    class ClickFunctionWrapper(t.Generic[P, T]):
        __name__: str
        __click_params__: t.List[click.Option]

        params: t.List[click.Parameter]

        def __call__(*args: P.args, **kwargs: P.kwargs) -> t.Callable[P, T]:
            ...

    WrappedCLI = t.Callable[P, ClickFunctionWrapper[t.Any, t.Any]]


@attrs.define
class Environment:

    _fs: FS = attrs.field(init=False)
    _manifest_dir: FS = attrs.field(init=False)
    _templates_dir: FS = attrs.field(init=False)
    _generated_dir: FS = attrs.field(init=False)

    # boolean
    overwrite: bool = False

    # misc
    organization: t.Optional[str] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("bentoml")
    )
    bentoml_version: t.Optional[str] = attrs.field(
        default=None, converter=attrs.converters.default_if_none(version("bentoml"))
    )
    cuda_version: t.Optional[str] = attrs.field(
        default=None, converter=attrs.converters.default_if_none("11.5.1")
    )
    python_version: t.Optional[t.Iterable[str]] = attrs.field(
        default=None,
        converter=attrs.converters.default_if_none(SUPPORTED_PYTHON_VERSION),
    )

    docker_package: t.Optional[str] = attrs.field(
        default=None,
        converter=attrs.converters.default_if_none(DockerManagerContainer.default_name),
    )

    distros: t.Optional[t.Iterable[str]] = attrs.field(
        default=None,
        converter=attrs.converters.default_if_none(SUPPORTED_OS_RELEASES),
    )

    build_ctx: t.Dict[str, t.List[BuildCtx]] = attrs.field(
        factory=dict,
        validator=attrs.validators.deep_mapping(
            attrs.validators.in_(SUPPORTED_OS_RELEASES),
            attrs.validators.deep_iterable(attrs.validators.instance_of(BuildCtx)),
        ),
    )
    release_ctx: t.Dict[str, t.List[ReleaseCtx]] = attrs.field(
        factory=dict,
        validator=attrs.validators.deep_mapping(
            attrs.validators.in_(SUPPORTED_OS_RELEASES),
            attrs.validators.deep_iterable(attrs.validators.instance_of(ReleaseCtx)),
        ),
    )

    @inject
    def __attrs_post_init__(
        self, root_fs: "FS" = Provide[DockerManagerContainer.root_fs]
    ):
        self._fs = root_fs
        self._manifest_dir = root_fs.makedirs("manifest", recreate=True)
        self._templates_dir = root_fs.makedirs("templates", recreate=True)
        self._generated_dir = root_fs.makedirs("generated", recreate=True)


pass_environment = click.make_pass_decorator(Environment, ensure=True)

if TYPE_CHECKING:
    ManagerWrapperCLI = WrappedCLI[
        Environment,
        bool,
        t.Optional[str],
        t.Optional[str],
        t.Optional[str],
        t.Optional[str],
        t.Optional[t.Iterable[str]],
        t.Optional[t.Iterable[str]],
    ]
    AnyWrapperCLI = ClickFunctionWrapper[t.Any, t.Any]


class ManagerCommandGroup(click.Group):
    COMMON_PARAMS = 7

    @staticmethod
    def common_params(
        func: AnyWrapperCLI, cmd_groups: click.Group
    ) -> ManagerWrapperCLI:
        @click.option(
            "--overwrite",
            required=False,
            is_flag=True,
            type=bool,
            help="whether to overwrite the manifest file.",
            default=False,
        )
        @click.option(
            "--organization",
            required=False,
            metavar="<bentoml>",
            type=click.STRING,
            default="bentoml",
            help="Targets organization, default is `bentoml` [optional]",
        )
        @click.option(
            "--docker-package",
            metavar="<package>",
            required=False,
            type=click.STRING,
            help="Target docker packages to use, default to `bento-server` [optional]",
        )
        @click.option(
            "--bentoml-version",
            required=False,
            metavar="<bentoml_version>",
            type=click.STRING,
            help="Targets bentoml version.",
        )
        @click.option(
            "--cuda-version",
            required=False,
            type=click.STRING,
            metavar="<x.y.z>",
            help="Targets CUDA version",
        )
        @click.option(
            "--python-version",
            required=False,
            type=click.Choice(SUPPORTED_PYTHON_VERSION),
            multiple=True,
            help=f"Targets a python version, default to {SUPPORTED_PYTHON_VERSION}",
        )
        @click.option(
            "--distros",
            required=False,
            type=click.Choice(SUPPORTED_OS_RELEASES),
            multiple=True,
            help="Targets a distros releases",
        )
        @wraps(func)
        @pass_environment
        @inject
        def wrapper(
            ctx: Environment,
            overwrite: bool,
            cuda_version: t.Optional[str],
            bentoml_version: t.Optional[str],
            organization: t.Optional[str],
            docker_package: t.Optional[str],
            distros: t.Optional[t.Iterable[str]],
            python_version: t.Optional[t.Iterable[str]],
            default_context: GenericDict = Provide[
                DockerManagerContainer.default_context
            ],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> t.Any:
            func_name = func.__name__.replace("_", "-")
            current_click_ctx = click.get_current_context()

            def get_options(name: str) -> click.Parameter:
                for opt in cmd.params:
                    if opt.name == name:
                        return opt
                raise ClickException(f"There is no Option {name}")

            if func_name not in ["auth-docker", "create-manifest"]:
                cmd = cmd_groups.commands[func_name]
                version_opt = get_options("bentoml_version")

                if bentoml_version is None:
                    raise click.MissingParameter(
                        ctx=current_click_ctx,
                        param=version_opt,
                    )

            from dotenv import load_dotenv

            _ = load_dotenv(dotenv_path=ctx._fs.getsyspath(".env"))

            if overwrite:
                ctx.overwrite = True

            if not python_version:
                python_version = SUPPORTED_PYTHON_VERSION
            if not distros:
                distros = SUPPORTED_OS_RELEASES

            ctx.cuda_version = cuda_version
            ctx.docker_package = docker_package
            ctx.bentoml_version = bentoml_version
            ctx.python_version = python_version

            if docker_package and docker_package != DockerManagerContainer.default_name:
                if cuda_version is None:
                    raise ManagerException(
                        "--cuda-version must be passed with --docker-package"
                    )
                get_distro_info = get_manifest_info(
                    docker_package=docker_package,
                    cuda_version=cuda_version,
                )
            else:
                get_distro_info = default_context  # type: ignore

            for k, v in get_distro_info.copy().items():
                get_distro_info[k] = {
                    key: attr for key, attr in v.items() if key in distros
                }

            ctx.distros = distros
            ctx.organization = organization

            set_generation_context(ctx, get_distro_info)
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def raise_click_exception(
        func: ManagerWrapperCLI, cmd_group: click.Group, **kwargs: t.Any
    ) -> AnyWrapperCLI:
        command_name = kwargs.get("name", func.__name__)

        @wraps(func)
        @raise_exception
        def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
            try:
                return func(*args, **kwargs)
            except ManagerException as err:
                msg = f"[{cmd_group.name}] `{command_name}` failed: {str(err)}"
                raise ClickException(click.style(msg, fg="red")) from err

        return t.cast("ClickFunctionWrapper[t.Any, t.Any]", wrapper)

    def command(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Callable[[AnyWrapperCLI], click.Command]:
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 120

        def wrapper(func: AnyWrapperCLI) -> click.Command:
            # add common parameters to command.
            wrapped = ManagerCommandGroup.common_params(func, self)
            # If ManagerException raise ClickException instead before exit.
            wrapped = ManagerCommandGroup.raise_click_exception(wrapped, self, **kwargs)

            # move common parameters to end of the parameters list
            wrapped.__click_params__ = (
                wrapped.__click_params__[-self.COMMON_PARAMS :]
                + wrapped.__click_params__[: -self.COMMON_PARAMS]
            )
            return super(ManagerCommandGroup, self).command(*args, **kwargs)(wrapped)

        return wrapper

    def format_help_text(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        sio = io.StringIO()
        self.console = Console(file=sio, force_terminal=True)
        self.console.print(self.__doc__)
        formatter.write(sio.getvalue())
