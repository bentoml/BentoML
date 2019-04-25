import os
import shutil
import subprocess
import argparse

from bentoml.utils import Path
from ruamel.yaml import YAML
from packaging import version

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

GOOGLE_MAIN_PY_TEMPLATE = """\
import {class_name}

bento_service = {class_name}.load()

def {api_name}(request):
    result = bento_service.{api_name}.handle_request(request)
    return result
"""

default_serverless_parser = argparse.ArgumentParser()
default_serverless_parser.add_argument('--region')
default_serverless_parser.add_argument('--stage')


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

    package_config = {'include': ['handler.py', bento_service.name + '/*', 'requirements.txt']}
    function_config = {
        'handler': 'handler.predict',
        'events': [{
            'http': {
                'path': '/predict',
                'method': 'post'
            }
        }]
    }

    serverless_config['package'] = package_config
    serverless_config['functions'][api.name] = function_config
    del serverless_config['functions']['hello']

    yaml.dump(serverless_config, Path(output_path))
    return


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


def generate_main_py(bento_service, output_path):
    main_file = os.path.join(output_path, 'main.py')
    api = bento_service.get_service_apis()[0]
    main_py_content = GOOGLE_MAIN_PY_TEMPLATE.format(class_name=bento_service.name,
                                                     api_name=api.name)

    with open(main_file, 'w') as f:
        f.write(main_py_content)
    return


def check_serverless_compatiable_version():
    version_result = subprocess.check_output(['serverless', '-v'])
    parsed_version = version.parse(version_result.decode('utf-8').strip())

    if parsed_version >= version.parse('1.40.0'):
        return
    else:
        raise ValueError(
            'Incompatiable serverless version, please install version 1.40.0 or greater')


def generate_serverless_bundle(bento_service, platform, archive_path, output_path, extra_args):
    TEMP_FOLDER_PATH = './temp_bento_serverless'
    check_serverless_compatiable_version()
    parsed_extra_args = default_serverless_parser.parse_args(extra_args)

    # Because Serverless framework will modify even absolute path with CWD.
    # So, if user provide an absolute path as parameter, we will generate the serverless
    # in the current directory and after our modification we will copy to user's desired
    # path and delete the temporary one we created.
    if os.path.isabs(output_path):
        is_absolute_path = True
        original_output_path = output_path
        output_path = TEMP_FOLDER_PATH
    else:
        is_absolute_path = False

    serverless_config_file = os.path.join(output_path, 'serverless.yml')
    generate_base_serverless_files(output_path, platform, bento_service.name)

    if platform != 'google-python':
        update_serverless_configuration_for_aws(bento_service, serverless_config_file,
                                                parsed_extra_args)
        generate_handler_py(bento_service, output_path)
    else:
        update_serverless_configuration_for_google(bento_service, serverless_config_file,
                                                   parsed_extra_args)
        generate_main_py(bento_service, output_path)

    shutil.copy(os.path.join(archive_path, 'requirements.txt'), output_path)
    add_model_service_archive(bento_service, archive_path, output_path)

    if is_absolute_path:
        shutil.copytree(output_path, original_output_path)
        shutil.rmtree(output_path)

    return
