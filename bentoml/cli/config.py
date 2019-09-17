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

from bentoml import config as bentoml_config
from bentoml.configuration import get_local_config_file, DEFAULT_CONFIG_FILE
from bentoml.cli.click_utils import _echo, CLI_COLOR_ERROR
from bentoml.utils.usage_stats import track_cli

# pylint: disable=unused-variable

LOG = logging.getLogger(__name__)

EXAMPLE_CONFIG_USAGE = '''
Example usage for `bentoml config`:
  bentoml config set usage_tracking=false
  bentoml config set apiserver.default_port=9000
  bentoml config unset apiserver.default_port
'''


def create_local_config_file_if_not_found():
    local_config_file = get_local_config_file()
    if not os.path.isfile(local_config_file):
        LOG.info("Creating new default BentoML config file at: %s", local_config_file)
        shutil.copyfile(DEFAULT_CONFIG_FILE, local_config_file)


def get_configuration_sub_command():
    @click.group(
        help="Configure BentoML configurations and settings",
        short_help="Config BentoML library",
    )
    def config():
        create_local_config_file_if_not_found()

    @config.command(help="View BentoML configurations")
    def view():
        track_cli('config-view')
        local_config = ConfigParser()
        with open(get_local_config_file(), 'rb') as config_file:
            local_config.read_string(config_file.read().decode('utf-8'))

        local_config.write(sys.stdout)
        return

    @config.command()
    def view_effective():
        track_cli('config-view-effective')
        bentoml_config().write(sys.stdout)
        return

    @config.command(
        name="set",
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
        help="Set value to BentoML configuration",
    )
    @click.argument("updates", nargs=-1)
    def set_command(updates):
        track_cli('config-set')
        local_config = ConfigParser()
        local_config_file = get_local_config_file()
        with open(local_config_file, 'rb') as config_file:
            local_config.read_string(config_file.read().decode('utf-8'))
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

            local_config.write(open(local_config_file, 'w'))
            return
        except ValueError:
            _echo('Wrong config format: %s' % str(updates), CLI_COLOR_ERROR)
            _echo(EXAMPLE_CONFIG_USAGE)
            return

    @config.command(
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
        help="Unset value from BentoML configuration",
    )
    @click.argument("updates", nargs=-1)
    def unset(updates):
        track_cli('config-unset')
        local_config = ConfigParser()
        local_config_file = get_local_config_file()
        with open(local_config_file, 'rb') as config_file:
            local_config.read_string(config_file.read().decode('utf-8'))
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

            local_config.write(open(local_config_file, 'w'))
            return
        except ValueError:
            _echo('Wrong config format: %s' % str(updates), CLI_COLOR_ERROR)
            _echo(EXAMPLE_CONFIG_USAGE)
            return

    @config.command(help="Reset BentoML configuration to default")
    def reset():
        track_cli('config-reset')
        local_config_file = get_local_config_file()
        if os.path.isfile(local_config_file):
            LOG.info("Removing existing BentoML config file: %s", local_config_file)
            os.remove(local_config_file)
        create_local_config_file_if_not_found()
        return

    return config
