import os
import shutil
import subprocess
from collections import OrderedDict

from bentoml.utils import Path
from ruamel.yaml import YAML

WITH_SERVERLESS_PYTHON_REQUIREMENT_PLUGIN = """\
package:
  include:
    - 'handler.py'
    - '{class_name}/*'
    - 'requirements.txt'

custom:
  pythonRequirements:
    slime: true
    layer: true
    dockerizePip: true
    zip: true

functions:
  {api_name}:
    layers:
      - {Ref: PythonRequirementsLambdaLayer}
    handler: handler.predict
    events:
      - http:
          path: /predict
          method: post
"""

WITHOUT_SERVERLESS_PYTHON_REQUIREMENT_PLUGIN = """\
package:
  include:
    - '*.py'
    - '{class_name}/*'
    - 'requirements.txt'

functions:
  {api_name}:
    handler: handler.predict
    events:
      - http:
          path: /predict
          method: post
"""

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


def generate_base_serverless_files(output_path, platform, name):
    subprocess.call(
        ['serverless', 'create', '--template', platform, '--path', output_path, '--name', name])
    if platform != 'google-python':
        subprocess.call(['serverless', 'plugin', 'install', '-n', 'serverless-python-requirements'],
                        cwd=output_path)
    return


def add_model_service_archive(bento_service, archive_path, output_path):
    model_serivce_archive_path = os.path.join(output_path, bento_service.name)
    shutil.copytree(archive_path, model_serivce_archive_path)
    return


def generate_handler_py(bento_service, output_path):
    handler_file = os.path.join(output_path, 'handler.py')
    api = bento_service.get_service_apis()[0]
    handler_py_content = AWS_HANDLER_PY_TEMPLATE.format(class_name=bento_service.name,
                                                        api_name=api.name)

    with open(handler_file, 'w') as f:
        f.write(handler_py_content)
    return


def update_serverless_configuration_for_aws(bento_service, output_path):
    yaml = YAML()
    api = bento_service.get_service_apis()[0]
    with open(output_path, 'r') as f:
        content = f.read()
    serverless_config = yaml.load(content)

    package_config = { 
        'include': ['handler.py', bento_service.name + '/*', 'requirements.txt']
    }
    function_config = {
        'handler': 'handler.predict',
        'events': [
            {
                'http': {
                    'path': '/predict',
                    'method': 'post'
                }
            }
        ]
    }

    serverless_config['package'] = package_config
    serverless_config['functions'][api.name] = function_config
    del serverless_config['functions']['hello']

    yaml.dump(serverless_config, Path(output_path))
    return


def generate_serverless_bundle(bento_service, platform, archive_path, output_path):
    serverless_config_file = os.path.join(output_path, 'serverless.yml')
    generate_base_serverless_files(output_path, platform, bento_service.name)

    if platform != 'google-python':
        update_serverless_configuration_for_aws(bento_service, serverless_config_file)
        generate_handler_py(bento_service, output_path)
    else:
        raise NotImplementedError

    shutil.copy(os.path.join(archive_path, 'requirements.txt'), output_path)
    add_model_service_archive(bento_service, archive_path, output_path)
    return
