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

import logging
import os

from ruamel.yaml import YAML

from bentoml.utils import Path

logger = logging.getLogger(__name__)

AWS_HANDLER_PY_TEMPLATE_HEADER = """\
try:
    import unzip_requirements:
except ImportError:
    pass

import {class_name}

bento_service = {class_name}.load()

"""

AWS_FUNCTION_TEMPLATE = """
def {api_name}(event, context):
    result = bento_service.{api_name}.handle_aws_lambda_event(event)

    return result

"""


def generate_serverless_configuration_for_aws(apis, output_path, additional_options):
    config_path = os.path.join(output_path, 'serverless.yml')
    yaml = YAML()
    with open(config_path, 'r') as f:
        content = f.read()
    serverless_config = yaml.load(content)

    if additional_options.get('region', None):
        serverless_config['provider']['region'] = additional_options['region']
        logger.info(('Using user defined AWS region: {0}', additional_options['region']))
    else:
        serverless_config['provider']['region'] = 'us-west-2'

    if additional_options.get('stage', None):
        serverless_config['provider']['stage'] = additional_options['stage']
        logger.info(('Using user defined AWS stage: {0}', additional_options['stage']))
    else:
        serverless_config['provider']['stage'] = 'dev'

    serverless_config['functions'] = {}
    for api in apis:
        function_config = {
            'handler': 'handler.{name}'.format(name=api.name),
            'layers': [
                '{Ref: PythonRequirementsLambdaLayer}'
            ],
            'events': [{
                'http': {
                    'path': '/{name}'.format(name=api.name),
                    'method': 'post'
                }
            }]
        }
        serverless_config['functions'][api.name] = function_config

    custom_config = {
        'apigwBinary': ['image/jpg', 'image/jpeg', 'image/png'],
        'pythonRequirements': {
            'useDownloadCache': True,
            'useStaticCache': True,
            'dockerizePip': True,
            'layer': True,
            'zip': True
        }
    }

    serverless_config['custom'] = custom_config

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_handler_py(bento_service, apis, output_path):
    handler_py_content = AWS_HANDLER_PY_TEMPLATE_HEADER.format(class_name=bento_service.name)

    for api in apis:
        api_content = AWS_FUNCTION_TEMPLATE.format(api_name=api.name)
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, 'handler.py'), 'w') as f:
        f.write(handler_py_content)
    return


def create_aws_lambda_bundle(bento_service, output_path, additional_options):
    apis = bento_service.get_service_apis()
    generate_handler_py(bento_service, apis, output_path)
    generate_serverless_configuration_for_aws(apis, output_path, additional_options)
    return
