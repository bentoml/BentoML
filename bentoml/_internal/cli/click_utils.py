import os
import re
import time
import typing as t
import difflib
import logging
import functools
from typing import TYPE_CHECKING

import click
from click import ClickException
from click.exceptions import UsageError

from ...exceptions import BentoMLException
from ..configuration import CONFIG_ENV_VAR
from ..configuration import set_debug_mode
from ..configuration import load_global_config
from ..utils.analytics import track
from ..utils.analytics import CliEvent
from ..utils.analytics import cli_events_map
from ..utils.analytics import BENTOML_DO_NOT_TRACK

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    class ClickFunctionWrapper(t.Protocol[P]):
        __name__: str
        __click_params__: t.List[click.Option]

        def __call__(*args: P.args, **kwargs: P.kwargs) -> t.Callable[P, t.Any]:
            ...

    WrappedCLI = t.Callable[P, ClickFunctionWrapper[t.Any]]


logger = logging.getLogger(__name__)


class BentoMLCommandGroup(click.Group):
    """Click command class customized for BentoML CLI, allow specifying a default
    command for each group defined
    """

    NUMBER_OF_COMMON_PARAMS = 4

    @staticmethod
    def bentoml_common_params(
        func: "t.Callable[P, t.Any]",
    ) -> "WrappedCLI[bool, bool, t.Optional[str]]":
        # NOTE: update NUMBER_OF_COMMON_PARAMS when adding option.

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
            envvar=BENTOML_DO_NOT_TRACK,
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
    ):
        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args: "P.args", **kwargs: "P.kwargs") -> t.Any:
            if do_not_track:
                os.environ[BENTOML_DO_NOT_TRACK] = str(True)
                return func(*args, **kwargs)

            start_time = time.time_ns()

            if (
                cmd_group.name in cli_events_map
                and command_name in cli_events_map[cmd_group.name]
            ):
                get_tracking_event = functools.partial(
                    cli_events_map[cmd_group.name][command_name],
                    cmd_group.name,
                    command_name,
                )
            else:

                def get_tracking_event(ret: t.Any) -> CliEvent:
                    return CliEvent(
                        cmd_group=cmd_group.name,
                        cmd_name=command_name,
                    )

            try:
                return_value = func(*args, **kwargs)
                event = get_tracking_event(return_value)
                duration_in_ns = time.time_ns() - start_time
                event.duration_in_ms = duration_in_ns / 1e6
                track(event)
                return return_value
            except BaseException as e:
                event = get_tracking_event(None)
                duration_in_ns = time.time_ns() - start_time
                event.duration_in_ms = duration_in_ns / 1e6
                event.error_type = type(e).__name__
                event.return_code = 2 if isinstance(e, KeyboardInterrupt) else 1
                track(event)
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
            func = BentoMLCommandGroup.raise_click_exception(func, self, **kwargs)

            # move common parameters to end of the parameters list
            func.__click_params__ = (
                func.__click_params__[-self.NUMBER_OF_COMMON_PARAMS :]
                + func.__click_params__[: -self.NUMBER_OF_COMMON_PARAMS]
            )
            return super(BentoMLCommandGroup, self).command(*args, **kwargs)(func)

        return wrapper

    def resolve_command(
        self, ctx: click.Context, args: t.List[str]
    ) -> t.Tuple[str, click.Command, t.List[str]]:
        try:
            return super(BentoMLCommandGroup, self).resolve_command(ctx, args)
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


def is_valid_bento_tag(value: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z_0-9]*:[A-Za-z0-9.+-_]*$", value) is not None


def is_valid_bento_name(value: str) -> bool:
    return re.match(r"^[A-Za-z_0-9]*$", value) is not None
