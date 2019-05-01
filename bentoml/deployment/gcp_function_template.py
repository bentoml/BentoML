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

import os
import logging

from ruamel.yaml import YAML

from bentoml.utils import Path

logger = logging.getLogger(__name__)

GOOGLE_MAIN_PY_TEMPLATE_HEADER = """\
import {class_name}

bento_service = {class_name}.load()

"""

GOOGLE_FUNCTION_TEMPLATE = """
def {api_name}(request):
    result = bento_service.{api_name}.handle_request(request)
    return result

"""


def generate_serverless_configuration_for_google(bento_service, apis, output_path,
                                                 additional_options):
    config_path = os.path.join(output_path, 'serverless.yml')
    yaml = YAML()
    with open(config_path, 'r') as f:
        content = f.read()
    serverless_config = yaml.load(content)
    serverless_config['provider']['project'] = bento_service.name

    if additional_options.get('region', None):
        serverless_config['provider']['region'] = additional_options['region']
        logger.info(('Using user defined Google region: {0}', additional_options['region']))
    if additional_options.get('stage', None):
        serverless_config['provider']['stage'] = additional_options['stage']
        logger.info(('Using user defined Google stage: {0}', additional_options['stage']))

    serverless_config['functions'] = {}
    for api in apis:
        if api.name == 'first':
            user_function_with_first_name = True

        function_config = {'handler': api.name, 'events': [{'http': 'path'}]}
        serverless_config['functions'][api.name] = function_config

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_main_py(bento_service, apis, output_path):
    handler_py_content = GOOGLE_MAIN_PY_TEMPLATE_HEADER.format(class_name=bento_service.name)

    for api in apis:
        api_content = GOOGLE_FUNCTION_TEMPLATE.format(api_name=api.name)
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, 'main.py'), 'w') as f:
        f.write(handler_py_content)
    return


def create_gcp_function_bundle(bento_service, output_path, additional_options):
    apis = bento_service.get_service_apis()
    generate_main_py(bento_service, apis, output_path)
    generate_serverless_configuration_for_google(bento_service, apis, output_path,
                                                 additional_options)
    return
