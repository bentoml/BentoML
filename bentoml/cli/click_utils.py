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
import click
import functools
import logging

from ruamel.yaml import YAML

from bentoml.utils.log import configure_logging

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


class BentoMLCommandGroup(click.Group):
    """Click command class customized for BentoML cli, allow specifying a default
    command for each group defined
    """

    NUMBER_OF_COMMON_PARAMS = 2

    @staticmethod
    def bentoml_common_params(func):
        @click.option(
            '-q',
            '--quiet',
            is_flag=True,
            default=False,
            help="Hide process logs and errors",
        )
        @click.option(
            '--verbose',
            '--debug',
            is_flag=True,
            default=False,
            help="Show additional details when running command",
        )
        @functools.wraps(func)
        def wrapper(quiet, verbose, *args, **kwargs):
            if verbose:
                from bentoml import config

                config().set('core', 'debug', 'true')
                configure_logging(logging.DEBUG)
            elif quiet:
                configure_logging(logging.ERROR)
            else:
                configure_logging()  # use default setting in local bentoml.cfg

            return func(*args, **kwargs)

        return wrapper

    def command(self, *args, **kwargs):
        def wrapper(func):
            # add common parameters to command
            func = BentoMLCommandGroup.bentoml_common_params(func)

            # move common parameters to end of the parameters list
            func.__click_params__ = (
                func.__click_params__[-self.NUMBER_OF_COMMON_PARAMS :]
                + func.__click_params__[: -self.NUMBER_OF_COMMON_PARAMS]
            )
            return super(BentoMLCommandGroup, self).command(*args, **kwargs)(func)

        return wrapper


def conditional_argument(condition, *param_decls, **attrs):
    """
    Attaches an argument to the command only when condition is True
    """

    def decorator(f):
        if condition:
            f = click.argument(*param_decls, **attrs)(f)
        return f

    return decorator


def parse_bento_tag_callback(ctx, param, value):  # pylint: disable=unused-argument
    if re.match(r"^[A-Za-z_][A-Za-z_0-9]*:[A-Za-z0-9.+-_]*$", value) is None:
        raise click.BadParameter(
            "Bad formatting. Please present in BentoName:Version, for example "
            "iris_classifier:v1.2.0"
        )

    return value


def parse_yaml_file_callback(ctx, param, value):  # pylint: disable=unused-argument
    yaml = YAML()
    yml_content = value.read()
    try:
        return yaml.load(yml_content)
    except Exception:
        raise click.BadParameter(
            'Input value is not recognizable yaml file or yaml content'
        )
