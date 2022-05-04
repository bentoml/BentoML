import typing as t

import click


def unparse_click_params(
    params: t.Dict[str, click.Parameter],
    command_params: t.List[click.Parameter],
) -> t.List[str]:
    """
    Unparse click call to a list of arguments. Used to modify some parameters and
    restore to system command.

    Args:
        ctx: click context

    Returns:
        List of arguments
    """
    args: t.List[str] = []
    for command_param in command_params:
        if isinstance(command_param, click.Argument):
            if command_param.name in params:
                args.append(str(params[command_param.name]))
        elif isinstance(command_param, click.Option):
            if command_param.name in params:
                if command_param.is_flag:
                    if params[command_param.name]:
                        args.append(command_param.opts[0])
                elif params[command_param.name] is not None:
                    args.append(f"--{command_param.name}".replace("_", "-"))
                    args.append(str(params[command_param.name]))
    return args
