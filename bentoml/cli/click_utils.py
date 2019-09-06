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

import re
import os
import click

from ruamel.yaml import YAML

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
from bentoml.exceptions import BentoMLException
from bentoml.utils import Path
from bentoml.utils.s3 import is_s3_url, download_from_s3
from bentoml.utils.tempdir import TempDirectory

CLI_COLOR_SUCCESS = "green"
CLI_COLOR_ERROR = "red"
CLI_COLOR_WARNING = "yellow"


def _echo(message, color="reset"):
    click.secho(message, fg=color)


class BentoMLCommandGroup(click.Group):
    """Click command class customized for BentoML cli, allow specifying a default
    command for each group defined
    """

    def command(self, *args, **kwargs):
        default_command = kwargs.pop("default_command", False)
        default_command_usage = kwargs.pop("default_command_usage", "")
        default_command_display_name = kwargs.pop("default_command_display_name", "<>")

        if default_command and not args:
            kwargs["name"] = kwargs.get("name", default_command_display_name)
        decorator = super(BentoMLCommandGroup, self).command(*args, **kwargs)

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
            return super(BentoMLCommandGroup, self).resolve_command(ctx, args)
        except click.UsageError:
            # command did not parse, assume it is the default command
            args.insert(0, self.default_command)
            return super(BentoMLCommandGroup, self).resolve_command(ctx, args)


def conditional_argument(condition, *param_decls, **attrs):
    """
    Attaches an argument to the command only when condition is True
    """

    def decorator(f):
        if condition:
            f = click.argument(*param_decls, **attrs)(f)
        return f

    return decorator


def parse_bento_tag_callback(ctx, param, value):
    if re.match(r"^[A-Za-z_][A-Za-z_0-9]*:[A-Za-z0-9.+-_]*$", value) is None:
        raise click.BadParameter(
            "Bad formatting. Please present in BentoName:Version, for example "
            "iris_classifier:v1.2.0"
        )

    return value


class TemporaryRemoteYamlFile(object):
    def __init__(self, remote_file_path, remote_storage_type='s3', _cleanup=True):
        if not remote_file_path.endswith(('.yml', '.yaml')):
            raise BentoMLException('Remote file must be YAML file.')
        self.remote_file_path = remote_file_path
        self.remote_storage_type = remote_storage_type
        self._cleanup = _cleanup
        self.temp_directory = TempDirectory()
        self.file_name = self.remote_file_path.split('/')[-1]
        self.file = None

    def __enter__(self):
        self.generate()
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup:
            self.cleanup()

    def generate(self):
        self.temp_directory.create()
        tempdir = self.temp_directory.path

        if self.remote_storage_type == 's3':
            download_from_s3(self.remote_file_path, tempdir)
            file_path = os.path.join(tempdir, self.file_name)
            self.file = open(file_path, 'rb')
            return self.file
        else:
            raise BentoMLException(
                'Remote storage {name} is not supported at the moment'.format(
                    name=self.remote_storage_type
                )
            )
        pass

    def cleanup(self):
        self.temp_directory.cleanup()
        self.file = None


def parse_yaml_file_or_string_callback(ctx, param, value):
    yaml = YAML()

    if os.path.isfile(Path(value)):
        with open(value, "rb") as yaml_file:
            yml_content = yaml_file.read()
    elif is_s3_url(value):
        with TemporaryRemoteYamlFile(value) as yaml_file:
            yml_content = yaml_file.read()
    else:
        yml_content = value
    try:
        result = yaml.load(yml_content)
        return result
    except Exception:
        raise click.BadParameter(
            'Input value is not recognizable yaml file or yaml content'
        )
