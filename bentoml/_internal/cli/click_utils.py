import os
import re
import time
import logging
import functools

import click
from click import ClickException

from ...exceptions import BentoMLException

# from bentoml import configure_logging
from ..configuration import CONFIG_ENV_VAR
from ..configuration import set_debug_mode
from ..configuration import load_global_config
from ..utils.usage_stats import track

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
TRACK_CLI_EVENT_NAME = "bentoml-cli"


def _echo(message, color="reset"):
    click.secho(message, fg=color)


class BentoMLCommandGroup(click.Group):
    """Click command class customized for BentoML cli, allow specifying a default
    command for each group defined
    """

    NUMBER_OF_COMMON_PARAMS = 4

    @staticmethod
    def bentoml_common_params(func):
        """Must update NUMBER_OF_COMMON_PARAMS above when adding or removing common CLI
        parameters here
        """

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
            envvar="BENTOML_DO_NOT_TRACK",
            help="Do not send usage info",
        )
        @click.option(
            "--config",
            type=click.Path(exists=True),
            envvar=CONFIG_ENV_VAR,
            help="BentoML configuration YAML file to apply",
        )
        @functools.wraps(func)
        def wrapper(quiet, verbose, config, *args, **kwargs):
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
    def bentoml_track_usage(func, cmd_group, **kwargs):
        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args, **kwargs):
            if do_not_track:
                os.environ["BENTOML_DO_NOT_TRACK"] = str(True)
                logger.debug(
                    "Executing '%s' command without usage tracking.", command_name
                )

            track_properties = {
                "command_group": cmd_group.name,
                "command": command_name,
            }
            start_time = time.time()
            try:
                return_value = func(*args, **kwargs)
                track_properties["duration"] = time.time() - start_time
                track_properties["return_code"] = 0
                track(TRACK_CLI_EVENT_NAME, track_properties)
                return return_value
            except BaseException as e:
                track_properties["duration"] = time.time() - start_time
                track_properties["error_type"] = type(e).__name__
                track_properties["error_message"] = str(e)
                track_properties["return_code"] = 1
                if type(e) == KeyboardInterrupt:
                    track_properties["return_code"] = 2
                track(TRACK_CLI_EVENT_NAME, track_properties)
                raise

        return wrapper

    @staticmethod
    def raise_click_exception(func, cmd_group, **kwargs):
        command_name = kwargs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BentoMLException as e:
                msg = f"{cmd_group.name} {command_name} failed: {str(e)}"
                raise ClickException(click.style(msg, fg="red"))

        return wrapper

    def command(self, *args, **kwargs):
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 120

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
