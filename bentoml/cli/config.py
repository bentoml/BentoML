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

import os
import sys
import click
import shutil
import logging

from configparser import ConfigParser

from bentoml.config import (
    config as bentoml_config,
    LOCAL_CONFIG_FILE,
    DEFAULT_CONFIG_FILE,
)
from bentoml.cli.click_utils import _echo, CLI_COLOR_ERROR

# pylint: disable=unused-variable

LOG = logging.getLogger(__name__)

EXAMPLE_CONFIG_USAGE = '''
Example usage for `bentoml config`:
  bentoml config set usage_tracking=false
  bentoml config set apiserver.default_port=9000
  bentoml config unset apiserver.default_port
'''


def create_local_config_file_if_not_found():
    if not os.path.isfile(LOCAL_CONFIG_FILE):
        LOG.info("Creating new default BentoML config file at: %s", LOCAL_CONFIG_FILE)
        shutil.copyfile(DEFAULT_CONFIG_FILE, LOCAL_CONFIG_FILE)


def add_configuration_commands(cli):
    @cli.command(
        help="Setting local default BentoML configurations",
        short_help="Config BentoML library",
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @click.argument(
        "action", type=click.Choice(["view", "view-effective", "set", "unset", "reset"])
    )
    @click.argument("updates", nargs=-1)
    def config(action, updates):
        create_local_config_file_if_not_found()

        if action == "view-effective":
            bentoml_config.write(sys.stdout)
            return

        if action == "reset":
            if os.path.isfile(LOCAL_CONFIG_FILE):
                LOG.info("Removing existing BentoML config file: %s", LOCAL_CONFIG_FILE)
                os.remove(LOCAL_CONFIG_FILE)
            create_local_config_file_if_not_found()
            return

        local_config = ConfigParser()
        with open(LOCAL_CONFIG_FILE, 'rb') as config_file:
            local_config.read_string(config_file.read().decode('utf-8'))

        if action == "view":
            local_config.write(sys.stdout)
        elif action == "set":
            try:
                for update in updates:
                    item, value = update.split('=')
                    if '.' in item:
                        sec, opt = item.split('.')
                    else:
                        sec = 'core'  # default section
                        opt = item

                    if not local_config.has_section(sec):
                        local_config.add_section(sec)
                    local_config.set(sec.strip(), opt.strip(), value.strip())

                local_config.write(open(LOCAL_CONFIG_FILE, 'w'))

            except ValueError:
                _echo('Wrong config format: %s' % str(updates), CLI_COLOR_ERROR)
                _echo(EXAMPLE_CONFIG_USAGE)
                return

        elif action == "unset":
            try:
                for update in updates:
                    if '.' in update:
                        sec, opt = update.split('.')
                    else:
                        sec = 'core'  # default section
                        opt = update

                    if not local_config.has_section(sec):
                        local_config.add_section(sec)
                    local_config.remove_option(sec.strip(), opt.strip())

                local_config.write(open(LOCAL_CONFIG_FILE, 'w'))

            except ValueError:
                _echo('Wrong config format: %s' % str(updates), CLI_COLOR_ERROR)
                _echo(EXAMPLE_CONFIG_USAGE)
                return
        else:
            LOG.error("Unknown command: bentoml config %s", action)
            return
        return

    return cli
