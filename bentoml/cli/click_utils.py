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

import re
import time
import os

import click
import functools
import logging

from click import ClickException

from bentoml import configure_logging
from bentoml.configuration import set_debug_mode
from bentoml.exceptions import BentoMLException
from bentoml.utils.ruamel_yaml import YAML
from bentoml.utils.usage_stats import track

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

logger = logging.getLogger(__name__)
TRACK_CLI_EVENT_NAME = 'bentoml-cli'


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
            help="Hide all warnings and info logs",
        )
        @click.option(
            '--verbose',
            '--debug',
            is_flag=True,
            default=False,
            help="Show debug logs when running the command",
        )
        @click.option(
            '--do-not-track',
            is_flag=True,
            default=False,
            envvar="BENTOML_DO_NOT_TRACK",
            help="Specify the option to not track usage.",
        )
        @functools.wraps(func)
        def wrapper(quiet, verbose, *args, **kwargs):
            if quiet:
                configure_logging(logging_level=logging.ERROR)
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
    def bentoml_track_usage(func, cmd_group, **kwargs):
        command_name = kwargs.get('name', func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args, **kwargs):
            if do_not_track:
                os.environ["BENTOML_DO_NOT_TRACK"] = str(True)
                logger.debug(
                    "Executing '%s' command without usage tracking.", command_name
                )

            track_properties = {
                'command_group': cmd_group.name,
                'command': command_name,
            }
            start_time = time.time()
            try:
                return_value = func(*args, **kwargs)
                track_properties['duration'] = time.time() - start_time
                track_properties['return_code'] = 0
                track(TRACK_CLI_EVENT_NAME, track_properties)
                return return_value
            except BaseException as e:
                track_properties['duration'] = time.time() - start_time
                track_properties['error_type'] = type(e).__name__
                track_properties['error_message'] = str(e)
                track_properties['return_code'] = 1
                if type(e) == KeyboardInterrupt:
                    track_properties['return_code'] = 2
                track(TRACK_CLI_EVENT_NAME, track_properties)
                raise

        return wrapper

    @staticmethod
    def raise_click_exception(func, cmd_group, **kwargs):
        command_name = kwargs.get('name', func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BentoMLException as e:
                msg = f'{cmd_group.name} {command_name} failed: {str(e)}'
                raise ClickException(click.style(msg, fg='red'))

        return wrapper

    def command(self, *args, **kwargs):
        def wrapper(func):
            # add common parameters to command
            func = BentoMLCommandGroup.bentoml_common_params(func)
            # Send tracking events before command finish.
            func = BentoMLCommandGroup.bentoml_track_usage(func, self, **kwargs)
            # If BentoMLException raise ClickException instead before exit
            func = BentoMLCommandGroup.raise_click_exception(func, self, **kwargs)

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


def _is_valid_bento_tag(value):
    return re.match(r"^[A-Za-z_][A-Za-z_0-9]*:[A-Za-z0-9.+-_]*$", value) is not None


def _is_valid_bento_name(value):
    return re.match(r"^[A-Za-z_0-9]*$", value) is not None


def parse_bento_tag_callback(ctx, param, value):  # pylint: disable=unused-argument
    if param.required and not _is_valid_bento_tag(value):
        raise click.BadParameter(
            "Bad formatting. Please present in BentoName:Version, for example "
            "iris_classifier:v1.2.0"
        )
    return value


def parse_labels_callback(ctx, param, value):  # pylint: disable=unused-argument
    if not value:
        return value

    parsed_labels = {}
    label_list = value.split(',')
    for label in label_list:
        if ':' not in label:
            raise click.BadParameter(
                f'Bad formatting for label {label}. '
                f'Please present label in key:value format'
            )
        label_key, label_value = label.split(':')
        parsed_labels[label_key] = label_value

    return parsed_labels


def validate_labels_query_callback(
    ctx, param, value
):  # pylint: disable=unused-argument
    if not value:
        return value

    labels = value.split(',')
    for label in labels:
        if '=' not in label:
            raise click.BadParameter(
                f'Bad formatting for label {label}. '
                f'Please present labels query in key=value format'
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
