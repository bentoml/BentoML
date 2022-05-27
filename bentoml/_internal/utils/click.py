from __future__ import annotations

import typing as t
import logging

import click

logger = logging.getLogger(__name__)


def unparse_click_params(
    params: dict[str, t.Any],
    command_params: list[click.Parameter],
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
