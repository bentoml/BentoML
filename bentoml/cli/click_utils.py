# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click


# Available CLI colors for _echo:
#
# _ansi_colors = {
#     'black': 30,
#     'red': 31,
#     'green': 32,
#     'yellow': 33,
#     'blue': 34,
#     'magenta': 35,
#     'cyan': 36,
#     'white': 37,
#     'reset': 39,
#     'bright_black': 90,
#     'bright_red': 91,
#     'bright_green': 92,
#     'bright_yellow': 93,
#     'bright_blue': 94,
#     'bright_magenta': 95,
#     'bright_cyan': 96,
#     'bright_white': 97,
# }
CLI_COLOR_SUCCESS = "green"
CLI_COLOR_ERROR = "red"
CLI_COLOR_WARNING = "yellow"


def _echo(message, color="reset"):
    click.secho(message, fg=color)


class DefaultCommandGroup(click.Group):
    """
    Allow a default command for a group, based on:
    """

    def command(self, *args, **kwargs):
        default_command = kwargs.pop("default_command", False)
        default_command_usage = kwargs.pop("default_command_usage", "")
        default_command_display_name = kwargs.pop("default_command_display_name", "<>")

        if default_command and not args:
            kwargs["name"] = kwargs.get("name", default_command_display_name)
        decorator = super(DefaultCommandGroup, self).command(*args, **kwargs)

        if default_command:

            def default_command_format_usage(ctx, formatter):
                formatter.write_usage(ctx.parent.command_path, default_command_usage)

            def new_decorator(f):
                cmd = decorator(f)
                cmd.format_usage = default_command_format_usage
                # pylint:disable=attribute-defined-outside-init
                self.default_command = cmd.name
                # pylint:enable=attribute-defined-outside-init

                return cmd

            return new_decorator

        return decorator

    def resolve_command(self, ctx, args):
        try:
            return super(DefaultCommandGroup, self).resolve_command(ctx, args)
        except click.UsageError:
            # command did not parse, assume it is the default command
            args.insert(0, self.default_command)
            return super(DefaultCommandGroup, self).resolve_command(ctx, args)


def conditional_argument(condition, *param_decls, **attrs):
    """
    Attaches an argument to the command only when condition is True
    """

    def decorator(f):
        if condition:
            f = click.argument(*param_decls, **attrs)(f)
        return f

    return decorator
