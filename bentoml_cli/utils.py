from __future__ import annotations

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

from bentoml.exceptions import BentoMLException
from bentoml._internal.log import configure_logging
from bentoml._internal.configuration import set_debug_mode
from bentoml._internal.configuration import set_quiet_mode
from bentoml._internal.utils.analytics import track
from bentoml._internal.utils.analytics import CliEvent
from bentoml._internal.utils.analytics import cli_events_map
from bentoml._internal.utils.analytics import BENTOML_DO_NOT_TRACK

if TYPE_CHECKING:
    from click import Option
    from click import Command
    from click import Context
    from click import Parameter

    P = t.ParamSpec("P")

    F = t.Callable[P, t.Any]

    class ClickFunctionWrapper(t.Protocol[P]):
        __name__: str
        __click_params__: list[Option]

        def __call__(  # pylint: disable=no-method-argument
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> F[P]:
            ...

    WrappedCLI = t.Callable[P, ClickFunctionWrapper[P]]


logger = logging.getLogger("bentoml")


class BentoMLCommandGroup(click.Group):
    """Click command class customized for BentoML CLI, allow specifying a default
    command for each group defined.
    """

    NUMBER_OF_COMMON_PARAMS = 3

    @staticmethod
    def bentoml_common_params(func: F[P]) -> WrappedCLI[bool, bool]:
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
        @functools.wraps(func)
        def wrapper(
            quiet: bool,
            verbose: bool,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> t.Any:
            if quiet:
                set_quiet_mode(True)
                if verbose:
                    logger.warning("'--quiet' passed; ignoring '--verbose/--debug'")
            elif verbose:
                set_debug_mode(True)

            configure_logging()

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def bentoml_track_usage(
        func: F[P] | WrappedCLI[bool, bool],
        cmd_group: click.Group,
        **kwargs: t.Any,
    ) -> WrappedCLI[bool]:
        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args: P.args, **kwargs: P.kwargs) -> t.Any:
            if do_not_track:
                os.environ[BENTOML_DO_NOT_TRACK] = str(True)
                return func(*args, **kwargs)

            start_time = time.time_ns()

            def get_tracking_event(return_value: t.Any) -> CliEvent:
                if TYPE_CHECKING:
                    # we only need to pass type checking here for group name
                    # since we know the cmd_group.name to be BentoMLCommandGroup
                    assert cmd_group.name

                if (
                    cmd_group.name in cli_events_map
                    and command_name in cli_events_map[cmd_group.name]
                ):
                    return cli_events_map[cmd_group.name][command_name](
                        cmd_group.name, command_name, return_value
                    )
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
        func: F[P] | WrappedCLI[bool], cmd_group: click.Group, **kwargs: t.Any
    ) -> ClickFunctionWrapper[t.Any]:
        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            try:
                return func(*args, **kwargs)
            except BentoMLException as err:
                msg = f"[{cmd_group.name}] `{command_name}` failed: {str(err)}"
                raise ClickException(click.style(msg, fg="red")) from err

        return t.cast("ClickFunctionWrapper[t.Any]", wrapper)

    def command(self, *args: t.Any, **kwargs: t.Any) -> t.Callable[[F[P]], Command]:
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 120

        def wrapper(func: F[P]) -> Command:
            # add common parameters to command.
            options = BentoMLCommandGroup.bentoml_common_params(func)
            # Send tracking events before command finish.
            usage = BentoMLCommandGroup.bentoml_track_usage(options, self, **kwargs)
            # If BentoMLException raise ClickException instead before exit.
            wrapped = BentoMLCommandGroup.raise_click_exception(usage, self, **kwargs)

            # move common parameters to end of the parameters list
            wrapped.__click_params__ = (
                wrapped.__click_params__[-self.NUMBER_OF_COMMON_PARAMS :]
                + wrapped.__click_params__[: -self.NUMBER_OF_COMMON_PARAMS]
            )
            return super(BentoMLCommandGroup, self).command(*args, **kwargs)(wrapped)

        return wrapper

    def resolve_command(
        self, ctx: Context, args: list[str]
    ) -> tuple[str | None, Command | None, list[str]]:
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
    try:
        from bentoml._internal.tag import Tag

        Tag.from_str(value)
        return True
    except ValueError:
        return False


def is_valid_bento_name(value: str) -> bool:
    return re.match(r"^[A-Za-z_0-9]*$", value) is not None


def unparse_click_params(
    params: dict[str, t.Any],
    command_params: list[Parameter],
    *,
    factory: t.Callable[..., t.Any] | None = None,
) -> list[str]:
    """
    Unparse click call to a list of arguments. Used to modify some parameters and
    restore to system command. The goal is to unpack cases where parameters can be parsed multiple times.

    Refers to ./buildx.py for examples of this usage. This is also used to unparse parameters for running API server.

    Args:
        params (`dict[str, t.Any]`):
            The dictionary of the parameters that is parsed from click.Context.
        command_params (`list[click.Parameter]`):
            The list of paramters (Arguments/Options) that is part of a given command.

    Returns:
        Unparsed list of arguments that can be redirected to system commands.

    Implementation:
        For cases where options is None, or the default value is None, we will remove it from the first params list.
        Currently it doesn't support unpacking `prompt_required` or `confirmation_prompt`.
    """
    args: list[str] = []

    # first filter out all parsed parameters that have value None
    # This means that the parameter was not set by the user
    params = {k: v for k, v in params.items() if v not in [None, (), []]}

    for command_param in command_params:
        if isinstance(command_param, click.Argument):
            # Arguments.nargs, Arguments.required
            if command_param.name in params:
                if command_param.nargs > 1:
                    # multiple arguments are passed as a list.
                    # In this case we try to convert all None to an empty string
                    args.extend(
                        list(
                            filter(
                                lambda x: "" if x is None else x,
                                params[command_param.name],
                            )
                        )
                    )
                else:
                    args.append(params[command_param.name])
        elif isinstance(command_param, click.Option):
            if command_param.name in params:
                if (
                    command_param.confirmation_prompt
                    or command_param.prompt is not None
                ):
                    logger.warning(
                        f"{command_params} is a prompt, skip parsing it for now."
                    )
                    pass
                if command_param.is_flag:
                    args.append(command_param.opts[-1])
                else:
                    cmd = f"--{command_param.name.replace('_','-')}"
                    if command_param.multiple:
                        for var in params[command_param.name]:
                            args.extend([cmd, var])
                    else:
                        args.extend([cmd, params[command_param.name]])
        else:
            logger.warning(
                "Given command params is a subclass of click.Parameter, but not a click.Argument or click.Option. Passing through..."
            )

    # We will also convert values if factory is parsed:
    if factory is not None:
        return list(map(factory, args))

    return args
