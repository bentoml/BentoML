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


def update_serverless_configuration_for_aws(bento_service, output_path, extra_args):
    yaml = YAML()
    api = bento_service.get_service_apis()[0]
    with open(output_path, 'r') as f:
        content = f.read()
    serverless_config = yaml.load(content)

    if extra_args.region:
        serverless_config['provider']['region'] = extra_args.region
    else:
        serverless_config['provider']['region'] = 'us-west-2'

    if extra_args.stage:
        serverless_config['provider']['stage'] = extra_args.stage
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
