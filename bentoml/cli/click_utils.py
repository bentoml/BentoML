# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click


class DefaultCommandGroup(click.Group):
    """
    Allow a default command for a group, based on:
    """

    def command(self, *args, **kwargs):
        default_command = kwargs.pop('default_command', False)
        default_command_usage = kwargs.pop('default_command_usage', '')
        default_command_display_name = kwargs.pop('default_command_display_name', '<>')

        if default_command and not args:
            kwargs['name'] = kwargs.get('name', default_command_display_name)
        decorator = super(DefaultCommandGroup, self).command(*args, **kwargs)

        if default_command:

            def default_command_format_usage(ctx, formatter):
                formatter.write_usage(ctx.parent.command_path, default_command_usage)

            def new_decorator(f):
                cmd = decorator(f)
                cmd.format_usage = default_command_format_usage
                self.default_command = cmd.name  # pylint:disable=attribute-defined-outside-init
                return cmd

            return new_decorator

        return decorator

    def resolve_command(self, ctx, args):
        try:
            # test if the command parses
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
