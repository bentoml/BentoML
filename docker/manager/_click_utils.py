import io
import os
import typing as t
import difflib
import logging
import traceback
from copy import deepcopy
from typing import TYPE_CHECKING
from functools import wraps

import attrs
import click
import attrs.converters
from click import ClickException
from toolz import dicttoolz
from fs.base import FS
from simple_di import inject
from simple_di import Provide
from rich.console import Console
from manager._make import set_generation_context
from manager._utils import SUPPORTED_REGISTRIES
from manager._utils import SUPPORTED_OS_RELEASES
from manager._utils import SUPPORTED_PYTHON_VERSION
from manager._utils import SUPPORTED_ARCHITECTURE_TYPE
from click.exceptions import UsageError
from manager._schemas import BuildCtx
from manager._schemas import ReleaseCtx
from plumbum.commands import ProcessExecutionError
from manager._exceptions import ManagerException
from manager._configuration import DockerRegistry
from manager._configuration import get_manifest_info
from manager._configuration import DockerManagerContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from manager._types import P
    from manager._types import WrappedCLI
    from manager._types import GenericDict
    from manager._types import ClickFunctionWrapper


CONTAINERSCRIPT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "containerscript")
)

OPTIONAL_CMD_FOR_VERSION = ["authenticate", "create-manifest", "push-readmes"]


def to_docker_targetarch(value: t.Optional[t.List[str]]) -> t.List[str]:
    result = []
    if not value:
        value = SUPPORTED_ARCHITECTURE_TYPE
    for x in value:
        if x == "arm64v8":
            x = "arm64"
        elif "arm32" in x:
            x = "arm"
        elif x == "i686":
            x = "386"
        elif x == "mips64":
            x = "mips64le"
        result.append(x)
    return result


@attrs.define
class Environment:

    _fs: FS = attrs.field(init=False)
    _manifest_dir: FS = attrs.field(init=False)
    _templates_dir: FS = attrs.field(init=False)
    _generated_dir: FS = attrs.field(init=False)

    # misc
    bentoml_version: str = attrs.field(
        default=None, converter=attrs.converters.default_if_none("")
    )
    cuda_version: str = attrs.field(
        default=None, converter=attrs.converters.default_if_none("11.5.1")
    )
    docker_package: str = attrs.field(
        default=None,
        converter=attrs.converters.default_if_none(DockerManagerContainer.default_name),
    )

    # boolean
    verbose: bool = attrs.field(default=True)
    quiet: bool = attrs.field(default=False)
    overwrite: bool = attrs.field(default=False)

    distros: t.Iterable[str] = attrs.field(
        converter=attrs.converters.default_if_none(SUPPORTED_OS_RELEASES),
        default=None,
        validator=lambda _, __, value: set(value).issubset(set(SUPPORTED_OS_RELEASES)),
    )
    python_version: t.Iterable[str] = attrs.field(
        converter=attrs.converters.default_if_none(SUPPORTED_PYTHON_VERSION),
        default=None,
        validator=lambda _, __, value: set(value).issubset(
            set(SUPPORTED_PYTHON_VERSION)
        ),
    )

    # ctx
    push_registry: t.Optional[str] = attrs.field(default=None)
    registries: t.Dict[str, DockerRegistry] = attrs.field(
        factory=dict,
        validator=attrs.validators.deep_mapping(
            attrs.validators.in_(SUPPORTED_REGISTRIES),
            attrs.validators.instance_of(DockerRegistry),
        ),
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

    # NOTE: we will use this for cross-platform helper.
    xx_version: str = attrs.field(
        default=None, converter=attrs.converters.default_if_none("1.1.0")
    )
    xx_image: str = attrs.field(
        default=None, converter=attrs.converters.default_if_none("tonistiigi/xx")
    )
    docker_target_arch: t.List[str] = attrs.field(
        default=None, converter=to_docker_targetarch
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
        bool,
        bool,
        str,
        str,
        str,
        t.Optional[str],
        t.Optional[t.Iterable[str]],
        t.Optional[t.Iterable[str]],
    ]
    AnyWrapperCLI = ClickFunctionWrapper[t.Any, t.Any]


class ManagerCommandGroup(click.Group):
    COMMON_PARAMS = 10

    @staticmethod
    def common_params(
        func: "AnyWrapperCLI", cmd_groups: click.Group
    ) -> "ManagerWrapperCLI":
        @click.option(
            "--bentoml-version",
            required=False,
            metavar="<bentoml_version>",
            type=click.STRING,
            help="Targets bentoml version.",
        )
        @click.option(
            "--docker-package",
            metavar="<package>",
            required=False,
            type=click.STRING,
            default=DockerManagerContainer.default_name,
            help="Target docker packages to use, default to `bento-server` [optional]",
        )
        @click.option(
            "--cuda-version",
            required=False,
            type=click.STRING,
            metavar="<x.y.z>",
            help="Targets CUDA version",
            default="11.5.1",
        )
        @click.option(
            "-q",
            "--quiet",
            is_flag=True,
            default=False,
            help="Suppress all warnings and info logs",
        )
        @click.option(
            "--verbose",
            is_flag=True,
            default=False,
            help="Generate debug information",
        )
        @click.option(
            "--overwrite",
            required=False,
            is_flag=True,
            type=bool,
            help="whether to overwrite the manifest file.",
            default=False,
        )
        @click.option(
            "--python-version",
            required=False,
            type=click.Choice(SUPPORTED_PYTHON_VERSION),
            multiple=True,
            help=f"Targets a python version, default to {SUPPORTED_PYTHON_VERSION}",
        )
        @click.option(
            "--registry",
            required=False,
            type=click.Choice(SUPPORTED_REGISTRIES),
            help="Targets registry to login.",
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
            quiet: bool,
            verbose: bool,
            overwrite: bool,
            cuda_version: str,
            docker_package: str,
            bentoml_version: str,
            registry: t.Optional[str],
            distros: t.Optional[t.Iterable[str]],
            python_version: t.Optional[t.Iterable[str]],
            *args: "P.args",
            default_context: "t.Tuple[GenericDict]" = Provide[
                DockerManagerContainer.default_context
            ],
            **kwargs: "P.kwargs",
        ) -> t.Any:
            func_name = func.__name__.replace("_", "-")
            current_click_ctx = click.get_current_context()
            create_manifest_func = cmd_groups.commands["create-manifest"]

            def get_options(name: str) -> click.Parameter:
                for opt in cmd.params:
                    if opt.name == name:
                        return opt
                raise ClickException(f"There is no Option {name}")

            if func_name not in OPTIONAL_CMD_FOR_VERSION:
                cmd = cmd_groups.commands[func_name]
                version_opt = get_options("bentoml_version")

                if bentoml_version is None:
                    raise click.MissingParameter(
                        ctx=current_click_ctx,
                        param=version_opt,
                    )

            from dotenv import load_dotenv

            _ = load_dotenv(dotenv_path=ctx._fs.getsyspath(".env"))

            if quiet:
                if verbose:
                    logger.warning("--verbose will be ignored when --quiet")
                logger.setLevel(logging.ERROR)
                ctx.verbose = False
            elif verbose:
                ctx.quiet = False
                logger.setLevel(logging.NOTSET)

            if overwrite:
                ctx.overwrite = True

            if not distros:
                distros = SUPPORTED_OS_RELEASES
            if not python_version:
                python_version = SUPPORTED_PYTHON_VERSION

            ctx.cuda_version = cuda_version
            ctx.python_version = python_version
            ctx.bentoml_version = bentoml_version

            if not ctx._manifest_dir.exists(DockerManagerContainer.default_manifest):
                current_click_ctx.forward(create_manifest_func)

            if docker_package != DockerManagerContainer.default_name:
                loaded_distros, loaded_registries = get_manifest_info(
                    docker_package=ctx.docker_package,
                    cuda_version=ctx.cuda_version,
                )
            else:
                loaded_distros, loaded_registries = default_context  # type: ignore

            if registry:
                registries = dicttoolz.keyfilter(
                    lambda x: x == registry, loaded_registries
                )
            else:
                registries = deepcopy(loaded_registries)

            ctx.distros = distros
            ctx.registries = registries
            ctx.push_registry = registry
            ctx.docker_package = docker_package

            set_generation_context(ctx, loaded_distros)

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def raise_click_exception(
        func: "ManagerWrapperCLI", cmd_group: click.Group, **kwargs: t.Any
    ) -> "AnyWrapperCLI":
        command_name = kwargs.get("name", func.__name__)

        @wraps(func)
        @pass_environment
        def wrapper(ctx: Environment, *args: t.Any, **kwargs: t.Any) -> t.Any:
            try:
                return func(*args, **kwargs)
            except (ManagerException, ProcessExecutionError) as err:
                if ctx.verbose:
                    msg = f"[{cmd_group.name}] `{command_name}` failed: {str(err)}"
                    raise ClickException(click.style(msg, fg="red")) from err
                elif ctx.quiet:
                    logger.info(
                        f"{command_name} failed while --quiet is passed. Remove --quiet to see the stack trace."
                    )
            except Exception:  # NOTE: for other exception show traceback
                logger.exception(traceback.format_exc())
                raise

        return t.cast("ClickFunctionWrapper[t.Any, t.Any]", wrapper)

    def command(
        self, *args: t.Any, **kwargs: t.Any
    ) -> "t.Callable[[AnyWrapperCLI], click.Command]":
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 120

        def wrapper(func: "AnyWrapperCLI") -> click.Command:
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

    def resolve_command(
        self, ctx: click.Context, args: t.List[str]
    ) -> t.Tuple[str, click.Command, t.List[str]]:
        try:
            return super(ManagerCommandGroup, self).resolve_command(ctx, args)
        except UsageError as e:
            error_msg = str(e)
            original_cmd_name = click.utils.make_str(args[0])
            matches = difflib.get_close_matches(
                original_cmd_name, self.list_commands(ctx), 3, 0.5
            )
            if matches:
                fmt_matches = "\n    ".join(matches)
                error_msg += "\n\n"
                error_msg += f"Did you mean?\n    {fmt_matches}"
            raise UsageError(error_msg, e.ctx)


class ContainerScriptGroup(ManagerCommandGroup):
    def list_commands(self, ctx: click.Context) -> t.Iterable[str]:
        rv = []
        for filename in os.listdir(CONTAINERSCRIPT_FOLDER):
            if filename.endswith(".py") and filename.startswith("reg_"):
                rv.append(filename[4:-3])
        rv.sort()
        return rv

    def get_command(self, ctx: click.Context, name: str) -> t.Optional[click.Command]:
        try:
            mod = __import__(
                f"manager.containerscript.reg_{name}", None, None, ["main"]
            )
        except ImportError:
            return
        return mod.main
