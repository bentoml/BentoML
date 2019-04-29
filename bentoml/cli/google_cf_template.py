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
    if extra_args.region:
        serverless_config['provider']['region'] = extra_args.region
    if extra_args.stage:
        serverless_config['provider']['stage'] = extra_args.stage
    serverless_config['provider']['project'] = bento_service.name

    function_config = {'handler': api.name, 'events': [{'http': 'path'}]}
    serverless_config['functions'][api.name] = function_config
    del serverless_config['functions']['first']
    yaml.dump(serverless_config, Path(output_path))
    return
