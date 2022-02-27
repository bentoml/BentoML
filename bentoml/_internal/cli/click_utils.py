import os
import re
import time
import typing as t
import logging
import functools
from typing import TYPE_CHECKING

import click
from click import ClickException

from ...exceptions import BentoMLException

# from ..log import configure_logging
from ..configuration import CONFIG_ENV_VAR
from ..configuration import set_debug_mode
from ..configuration import load_global_config
from ..utils.analytics import track
from ..utils.analytics import CLI_TRACK_EVENT_TYPE

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    class ClickFunctionWrapper(t.Protocol[P]):
        __name__: str
        __click_params__: t.List[click.Option]

        def __call__(*args: P.args, **kwargs: P.kwargs) -> t.Callable[P, t.Any]:
            ...

    WrappedCLI = t.Callable[P, ClickFunctionWrapper[t.Any]]


logger = logging.getLogger(__name__)
# Below contains commands that will have different tracking implementation.
CMD_WITH_CUSTOM_TRACKING = ["save", "build", "serve"]


# TODO: implement custom help message format
class BentoMLCommandGroup(click.Group):
    """Click command class customized for BentoML CLI, allow specifying a default
    command for each group defined
    """

    NUMBER_OF_COMMON_PARAMS = 4

    @staticmethod
    def bentoml_common_params(
        func: "t.Callable[P, t.Any]",
    ) -> "WrappedCLI[bool, bool, t.Optional[str]]":
        """Must update NUMBER_OF_COMMON_PARAMS above when adding or removing common CLI
        parameters here.
        """

        @click.option(
            "-q",
            "--quiet",
            is_flag=True,
            default=False,
            help="Suppress all warnings and info logs",
        )
        @click.option(
            "--verbose",
            "--debug",
            is_flag=True,
            default=False,
            help="Generate debug information",
        )
        @click.option(
            "--do-not-track",
            is_flag=True,
            default=False,
            envvar="BENTOML_DO_NOT_TRACK",
            help="Do not send usage info",
        )
        @click.option(
            "--config",
            type=click.Path(exists=True),
            envvar=CONFIG_ENV_VAR,
            help="BentoML configuration YAML file to apply",
        )
        @functools.wraps(func)
        def wrapper(
            quiet: bool,
            verbose: bool,
            config: t.Optional[str],
            *args: "P.args",
            **kwargs: "P.kwargs",
        ) -> t.Any:
            if config:
                load_global_config(config)

            if quiet:
                # TODO: fix configure logging
                # configure_logging(logging_level=logging.ERROR)
                if verbose:
                    logger.warning(
                        "The bentoml command option `--verbose/--debug` is ignored when"
                        "the `--quiet` flag is also in use"
                    )
            elif verbose:
                set_debug_mode(True)

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def bentoml_track_usage(
        func: t.Union["t.Callable[P, t.Any]", "ClickFunctionWrapper[t.Any]"],
        cmd_group: click.Group,
        **kwargs: t.Any,
    ) -> "WrappedCLI[bool]":
        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args: "P.args", **kwargs: "P.kwargs") -> t.Any:
            if do_not_track:
                os.environ["BENTOML_DO_NOT_TRACK"] = str(True)
                logger.debug(
                    "Executing '%s' command without usage tracking.", command_name
                )
            if command_name in CMD_WITH_CUSTOM_TRACKING:
                return func(*args, **kwargs)

            process_pid = os.getpid()
            cli_properties = {
                "command_group": cmd_group.name,
                "command": command_name,
                "error_message": "",
                "error_type": "",
            }
            start_time = time.time()
            try:
                return_value = func(*args, **kwargs)
                cli_properties["duration"] = time.time() - start_time
                cli_properties["return_code"] = 0
                track(
                    CLI_TRACK_EVENT_TYPE, process_pid, event_properties=cli_properties
                )
                return return_value
            except BaseException as e:
                cli_properties["duration"] = time.time() - start_time
                cli_properties["error_type"] = type(e).__name__
                cli_properties["error_message"] = str(e)
                cli_properties["return_code"] = 1
                if type(e) == KeyboardInterrupt:
                    cli_properties["return_code"] = 2
                track(
                    CLI_TRACK_EVENT_TYPE, process_pid, event_properties=cli_properties
                )
                raise

        return wrapper

    @staticmethod
    def raise_click_exception(
        func: t.Union["t.Callable[P, t.Any]", "ClickFunctionWrapper[t.Any]"],
        cmd_group: click.Group,
        **kwargs: t.Any,
    ) -> "ClickFunctionWrapper[t.Any]":
        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
            try:
                return func(*args, **kwargs)
            except BentoMLException as err:
                msg = f"[{cmd_group.name}] `{command_name}` failed: {str(err)}"
                raise ClickException(click.style(msg, fg="red")) from err

        return t.cast("ClickFunctionWrapper[t.Any]", wrapper)

    def command(
        self, *args: t.Any, **kwargs: t.Any
    ) -> "t.Callable[[t.Callable[P, t.Any]], click.Command]":
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 120

        def wrapper(func: "t.Callable[P, t.Any]") -> click.Command:
            # add common parameters to command.
            func = BentoMLCommandGroup.bentoml_common_params(func)
            # Send tracking events before command finish.
            func = BentoMLCommandGroup.bentoml_track_usage(func, self, **kwargs)
            # If BentoMLException raise ClickException instead before exit.
            # NOTE: Always call this function last for type checker to work.
            func = BentoMLCommandGroup.raise_click_exception(func, self, **kwargs)

            # move common parameters to end of the parameters list
            func.__click_params__ = (
                func.__click_params__[-self.NUMBER_OF_COMMON_PARAMS :]
                + func.__click_params__[: -self.NUMBER_OF_COMMON_PARAMS]
            )
            return super(BentoMLCommandGroup, self).command(*args, **kwargs)(func)

        return wrapper


def is_valid_bento_tag(value: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z_0-9]*:[A-Za-z0-9.+-_]*$", value) is not None


def is_valid_bento_name(value: str) -> bool:
    return re.match(r"^[A-Za-z_0-9]*$", value) is not None
