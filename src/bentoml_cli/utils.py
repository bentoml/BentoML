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

if TYPE_CHECKING:
    from click import Option
    from click import Command
    from click import Context
    from click import Parameter
    from click import HelpFormatter

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

    ClickParamType = t.Sequence[t.Any] | bool | None

logger = logging.getLogger("bentoml")


def kwargs_transformers(
    f: F[t.Any] | None = None,
    *,
    transformer: F[t.Any],
    pass_click_context: bool = False,
) -> F[t.Any]:
    def decorator(_f: F[t.Any]) -> t.Callable[P, t.Any]:
        @functools.wraps(_f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            transformed = {k: transformer(v) for k, v in kwargs.items()}
            if pass_click_context:
                return _f(click.get_current_context(), *args, **transformed)
            return _f(*args, **transformed)

        return wrapper

    if f is None:
        return decorator
    return decorator(f)


def _validate_docker_tag(tag: str) -> str:
    from bentoml.exceptions import BentoMLException

    if ":" in tag:
        name, version = tag.split(":")[:2]
    else:
        name, version = tag, None

    valid_name_pattern = re.compile(
        r"""
        ^(
        [a-z0-9]+      # alphanumeric
        (.|_{1,2}|-+)? # seperators
        )*$
        """,
        re.VERBOSE,
    )
    valid_version_pattern = re.compile(
        r"""
        ^
        [a-zA-Z0-9] # cant start with .-
        [ -~]{,127} # ascii match rest, cap at 128
        $
        """,
        re.VERBOSE,
    )

    if not valid_name_pattern.match(name):
        raise BentoMLException(
            f"Provided Docker Image tag {tag} is invalid. Name components may contain lowercase letters, digits and separators. A separator is defined as a period, one or two underscores, or one or more dashes."
        )
    if version and not valid_version_pattern.match(version):
        raise BentoMLException(
            f"Provided Docker Image tag {tag} is invalid. A tag name must be valid ASCII and may contain lowercase and uppercase letters, digits, underscores, periods and dashes. A tag name may not start with a period or a dash and may contain a maximum of 128 characters."
        )
    return tag


def validate_container_tag(
    ctx: Context, param: Parameter, tag: str | tuple[str] | None
) -> str | tuple[str] | None:
    from bentoml.exceptions import BentoMLException

    if not tag:
        return tag
    elif isinstance(tag, tuple):
        return tuple(map(_validate_docker_tag, tag))
    elif isinstance(tag, str):
        return _validate_docker_tag(tag)
    else:
        raise BentoMLException(f"Invalid tag type. Got {type(tag)}")


# NOTE: shim for bentoctl
def validate_docker_tag(
    ctx: Context, param: Parameter, tag: str | tuple[str] | None
) -> str | tuple[str] | None:
    logger.warning(
        "'validate_docker_tag' is now deprecated, use 'validate_container_tag' instead."
    )
    return validate_container_tag(ctx, param, tag)


@t.overload
def normalize_none_type(value: t.Mapping[str, t.Any]) -> t.Mapping[str, t.Any]:
    ...


@t.overload
def normalize_none_type(value: ClickParamType) -> ClickParamType:
    ...


def normalize_none_type(
    value: ClickParamType | t.Mapping[str, t.Any]
) -> ClickParamType | t.Mapping[str, t.Any]:
    if isinstance(value, (tuple, list, set, str)) and len(value) == 0:
        return
    if isinstance(value, dict):
        return {k: normalize_none_type(v) for k, v in value.items()}
    return value


def flatten_opt_tuple(value: t.Any) -> t.Any:
    from bentoml._internal.types import LazyType

    if LazyType["tuple[t.Any, ...]"](tuple).isinstance(value) and len(value) == 1:
        return value[0]
    return value


# NOTE: This is the key we use to store the transformed options in the CLI context.
MEMO_KEY = "_memoized"


def opt_callback(ctx: Context, param: Parameter, value: ClickParamType):
    # NOTE: our new options implementation will have the following format:
    #   --opt ARG[=|:]VALUE[,VALUE] (e.g., --opt key1=value1,value2 --opt key2:value3/value4:hello)
    # Argument and values per --opt has one-to-one or one-to-many relationship,
    # separated by '=' or ':'.
    # TODO: We might also want to support the following format:
    #  --opt arg1 value1 arg2 value2 --opt arg3 value3
    #  --opt arg1=value1,arg2=value2 --opt arg3:value3

    assert param.multiple, "Only use this callback when multiple=True."
    if MEMO_KEY not in ctx.params:
        # By default, multiple options are stored as a tuple.
        # our memoized options are stored as a dict.
        ctx.params[MEMO_KEY] = {}

    if param.name not in ctx.params[MEMO_KEY]:
        ctx.params[MEMO_KEY][param.name] = ()

    value = normalize_none_type(value)
    if value is not None and isinstance(value, tuple):
        for opt in value:
            o, *val = re.split(r"=|:", opt, maxsplit=1)
            norm = o.replace("-", "_")
            if len(val) == 0:
                # --opt bool
                ctx.params[MEMO_KEY][norm] = True
            else:
                # --opt key=value
                ctx.params[MEMO_KEY].setdefault(norm, ())
                ctx.params[MEMO_KEY][norm] += (*val,)
    return value


class BentoMLCommandGroup(click.Group):
    """
    Click command class customized for BentoML CLI, allow specifying a default
    command for each group defined.

    This command groups will also introduce support for aliases for commands.

    Example:

    .. code-block:: python

        @click.group(cls=BentoMLCommandGroup)
        def cli(): ...

        @cli.command(aliases=["serve-http"])
        def serve(): ...
    """

    NUMBER_OF_COMMON_PARAMS = 3

    @staticmethod
    def bentoml_common_params(func: F[P]) -> WrappedCLI[bool, bool]:
        # NOTE: update NUMBER_OF_COMMON_PARAMS when adding option.
        from bentoml._internal.log import configure_logging
        from bentoml._internal.configuration import DEBUG_ENV_VAR
        from bentoml._internal.configuration import QUIET_ENV_VAR
        from bentoml._internal.configuration import set_debug_mode
        from bentoml._internal.configuration import set_quiet_mode
        from bentoml._internal.utils.analytics import BENTOML_DO_NOT_TRACK

        @click.option(
            "-q",
            "--quiet",
            is_flag=True,
            default=False,
            envvar=QUIET_ENV_VAR,
            help="Suppress all warnings and info logs",
        )
        @click.option(
            "--verbose",
            "--debug",
            is_flag=True,
            default=False,
            envvar=DEBUG_ENV_VAR,
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
        from bentoml._internal.utils.analytics import track
        from bentoml._internal.utils.analytics import CliEvent
        from bentoml._internal.utils.analytics import cli_events_map
        from bentoml._internal.utils.analytics import BENTOML_DO_NOT_TRACK

        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args: P.args, **kwargs: P.kwargs) -> t.Any:
            if do_not_track:
                os.environ[BENTOML_DO_NOT_TRACK] = str(True)
                return func(*args, **kwargs)

            start_time = time.time_ns()

            def get_tracking_event(return_value: t.Any) -> CliEvent:
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
        from bentoml.exceptions import BentoMLException
        from bentoml._internal.configuration import get_debug_mode

        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            try:
                return func(*args, **kwargs)
            except BentoMLException as err:
                msg = f"[{cmd_group.name}] `{command_name}` failed: {str(err)}"
                if get_debug_mode():
                    ClickException(click.style(msg, fg="red")).show()
                    raise err from None
                else:
                    raise ClickException(click.style(msg, fg="red")) from err

        return t.cast("ClickFunctionWrapper[t.Any]", wrapper)

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super(BentoMLCommandGroup, self).__init__(*args, **kwargs)
        # these two dictionaries will store known aliases for commands and groups
        self._commands: dict[str, list[str]] = {}
        self._aliases: dict[str, str] = {}

    def command(self, *args: t.Any, **kwargs: t.Any) -> t.Callable[[F[P]], Command]:
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 120
        aliases = kwargs.pop("aliases", None)

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
            cmd = super(BentoMLCommandGroup, self).command(*args, **kwargs)(wrapped)
            # add aliases to a given commands if it is specified.
            if aliases is not None:
                assert cmd.name
                self._commands[cmd.name] = aliases
                self._aliases.update({alias: cmd.name for alias in aliases})
            return cmd

        return wrapper

    def resolve_alias(self, cmd_name: str):
        return self._aliases[cmd_name] if cmd_name in self._aliases else cmd_name

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:
        cmd_name = self.resolve_alias(cmd_name)
        return super(BentoMLCommandGroup, self).get_command(ctx, cmd_name)

    def format_commands(self, ctx: Context, formatter: HelpFormatter) -> None:
        rows: list[tuple[str, str]] = []
        sub_commands = self.list_commands(ctx)

        max_len = max(len(cmd) for cmd in sub_commands)
        limit = formatter.width - 6 - max_len

        for sub_command in sub_commands:
            cmd = self.get_command(ctx, sub_command)
            if cmd is None:
                continue
            # If the command is hidden, then we skip it.
            if hasattr(cmd, "hidden") and cmd.hidden:
                continue
            if sub_command in self._commands:
                aliases = ", ".join(sorted(self._commands[sub_command]))
                sub_command = "%s (%s)" % (sub_command, aliases)
            # this cmd_help is available since click>=7
            # BentoML requires click>=7.
            cmd_help = cmd.get_short_help_str(limit)
            rows.append((sub_command, cmd_help))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

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

    Args:
        params: The dictionary of the parameters that is parsed from click.Context.
        command_params: The list of paramters (Arguments/Options) that is part of a given command.

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
                        "%s is a prompt, skip parsing it for now.", command_params
                    )
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
