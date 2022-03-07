import io
import typing as t
import logging
from typing import TYPE_CHECKING
from functools import wraps
from functools import lru_cache

import click
from click import ClickException
from manager import __version__ as MANAGER_VERSION
from rich.console import Console
from manager.build import add_build_command
from manager._utils import graceful_exit
from manager.generate import add_generation_command
from manager.exceptions import ManagerException
from manager.authenticate import add_authenticate_command

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from manager._types import P
    from manager._types import WrappedCLI
    from manager._types import GenericFunc
    from manager._types import ClickFunctionWrapper


@lru_cache(maxsize=1)
def _rich_callback():
    import sys

    if sys.stdout.isatty():
        # add traceback when in interactive shell for development
        from rich.traceback import install

        install(suppress=[click])


class ManagerCommandGroup(click.Group):
    COMMON_PARAMS = 3

    @staticmethod
    def common_params(func: "GenericFunc[t.Any]") -> "WrappedCLI[bool, bool]":
        @click.option(
            "--docker-package",
            metavar="<package>",
            required=False,
            type=click.STRING,
            default="bento-server",
            help="Target docker packages to use, default to `bento-server` [optional]",
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
        @wraps(func)
        def wrapper(
            quiet: bool, verbose: bool, *args: "P.args", **kwargs: "P.kwargs"
        ) -> t.Any:
            if quiet:
                if verbose:
                    logger.warning("--verbose will be ignored when --quiet")
                logger.setLevel(logging.INFO)
            elif verbose:
                logger.setLevel(logging.NOTSET)
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def raise_click_exception(
        func: t.Union["GenericFunc[t.Any]", "WrappedCLI[bool, bool]"],
        cmd_group: click.Group,
        **kwargs: t.Any,
    ) -> "ClickFunctionWrapper[t.Any, t.Any]":
        command_name = kwargs.get("name", func.__name__)

        @wraps(func)
        def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
            try:
                return func(*args, **kwargs)
            except ManagerException as err:
                msg = f"[{cmd_group.name}] `{command_name}` failed: {str(err)}"
                raise ClickException(click.style(msg, fg="red")) from err

        return t.cast("ClickFunctionWrapper[t.Any, t.Any]", wrapper)

    def command(
        self, *args: t.Any, **kwargs: t.Any
    ) -> "t.Callable[[GenericFunc[t.Any]], click.Command]":
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 120

        def wrapper(func: "GenericFunc[t.Any]") -> click.Command:
            # add common parameters to command.
            wrapped = ManagerCommandGroup.common_params(func)
            # If BentoMLException raise ClickException instead before exit.
            # NOTE: Always call this function last for type checker to work.
            wrapped = ManagerCommandGroup.raise_click_exception(wrapped, self, **kwargs)

            # move common parameters to end of the parameters list
            wrapped.__click_params__ = (
                wrapped.__click_params__[-self.COMMON_PARAMS :]
                + wrapped.__click_params__[: -self.COMMON_PARAMS]
            )
            return super(ManagerCommandGroup, self).command(*args, **kwargs)(wrapped)

        return wrapper

    def format_help_text(
        self, ctx: "click.Context", formatter: "click.HelpFormatter"
    ) -> None:
        sio = io.StringIO()
        self.console = Console(file=sio, force_terminal=True)
        self.console.print(self.__doc__)
        formatter.write(sio.getvalue())


@graceful_exit
def create_manager_cli():
    _rich_callback()

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    # fmt: off
    @click.group(cls=ManagerCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(MANAGER_VERSION, "-v", "--version")
    def cli() -> None:
        """
[bold yellow]Manager[/bold yellow]: BentoML's Docker Images release management system.

[bold red]Features[/bold red]:

    :memo: Multiple Python version: 3.7, 3.8, 3.9+, ...
    :memo: Multiple platform: arm64v8, amd64, ppc64le, ...
    :memo: Multiple Linux Distros that you love: Debian, Ubuntu, UBI, alpine, ...

Get started with:
    $ manager --help
        """

    # fmt: on
    add_authenticate_command(cli)
    add_generation_command(cli)
    add_build_command(cli)

    return cli


cli = create_manager_cli()

if __name__ == "__main__":
    cli()
