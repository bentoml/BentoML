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


from ruamel.yaml import YAML

from bentoml.utils import Path

AWS_HANDLER_PY_TEMPLATE = """\
try:
    import unzip_requirements:
except ImportError:
    pass

import json

import {class_name}

bento_service = {class_name}.load()

def {api_name}(event, context):
    result = bento_service.{api_name}.handle_aws_lambda_event(event)

    return result
"""


def update_serverless_configuration_for_aws(bento_service, output_path, additional_options):
    yaml = YAML()
    api = bento_service.get_service_apis()[0]
    with open(output_path, 'r') as f:
        content = f.read()
    serverless_config = yaml.load(content)

    if additional_options.get('region', None):
        serverless_config['provider']['region'] = additional_options['region']
    else:
        serverless_config['provider']['region'] = 'us-west-2'

    if additional_options.get('stage', None):
        serverless_config['provider']['stage'] = additional_options['stage']
    else:
        serverless_config['provider']['stage'] = 'dev'

    function_config = {
        'handler': 'handler.predict',
        'events': [{
            'http': {
                'path': '/predict',
                'method': 'post'
            }
        }]
    }
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
    serverless_config['functions'][api.name] = function_config
    #del serverless_config['functions']['hello']

    yaml.dump(serverless_config, Path(output_path))
    return
