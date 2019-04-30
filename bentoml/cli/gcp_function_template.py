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

GOOGLE_MAIN_PY_TEMPLATE = """\
import {class_name}

bento_service = {class_name}.load()

def {api_name}(request):
    result = bento_service.{api_name}.handle_request(request)
    return result
"""


def update_serverless_configuration_for_google(bento_service, output_path, extra_args):
    yaml = YAML()
    api = bento_service.get_service_apis()[0]
    with open(output_path, 'r') as f:
        content = f.read()
    serverless_config = yaml.load(content)
    if extra_args.get('region', None):
        serverless_config['provider']['region'] = extra_args['region']
    if extra_args.get('stage', None):
        serverless_config['provider']['stage'] = extra_args['stage']
    serverless_config['provider']['project'] = bento_service.name

    function_config = {'handler': api.name, 'events': [{'http': 'path'}]}
    serverless_config['functions'][api.name] = function_config
    del serverless_config['functions']['first']
    yaml.dump(serverless_config, Path(output_path))
    return
