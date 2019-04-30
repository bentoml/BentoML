import os
import shutil
import subprocess
import argparse

from datetime import datetime
from packaging import version

from bentoml.cli.whichcraft import which
from bentoml.cli.aws_lambda_template import AWS_HANDLER_PY_TEMPLATE, \
     update_serverless_configuration_for_aws
from bentoml.cli.gcp_function_template import GOOGLE_MAIN_PY_TEMPLATE, \
     update_serverless_configuration_for_google

default_serverless_parser = argparse.ArgumentParser()
default_serverless_parser.add_argument('--region')
default_serverless_parser.add_argument('--stage')

SERVERLESS_PROVIDER = {
    'aws-lambda': 'aws-python',
    'aws-lambda-py3': 'aws-python3',
    'gcp-function': 'google-python',
}


def generate_base_serverless_files(output_path, platform, name):
    subprocess.call(
        ['serverless', 'create', '--template', platform, '--path', output_path, '--name', name])
    if platform != 'google-python':
        subprocess.call(['serverless', 'plugin', 'install', '-n', 'serverless-python-requirements'],
                        cwd=output_path)
        subprocess.call(['serverless', 'plugin', 'install', '-n', 'serverless-apigw-binary'],
                        cwd=output_path)
    return


def deploy_serverless_file(output_path):
    subprocess.call(['serverless', 'deploy'], cwd=output_path)
    return


def add_model_service_archive(bento_service, archive_path, output_path):
    model_serivce_archive_path = os.path.join(output_path, bento_service.name)
    shutil.copytree(archive_path, model_serivce_archive_path)
    return


def generate_handler_py(bento_service, output_path, platform):
    api = bento_service.get_service_apis()[0]
    if platform == 'google-python':
        file_name = 'main.py'
        handler_py_content = GOOGLE_MAIN_PY_TEMPLATE.format(class_name=bento_service.name,
                                                            api_name=api.name)
    else:
        file_name = 'handler.py'
        handler_py_content = AWS_HANDLER_PY_TEMPLATE.format(class_name=bento_service.name,
                                                            api_name=api.name)

    handler_file = os.path.join(output_path, file_name)

    with open(handler_file, 'w') as f:
        f.write(handler_py_content)
    return


def check_serverless_compatiable_version():
    if which('serverless') is None:
        raise ValueError(
            'Serverless framework is not installed, please visit ' +
            'www.serverless.com for install instructions.'
        )

    version_result = subprocess.check_output(['serverless', '-v'])
    parsed_version = version.parse(version_result.decode('utf-8').strip())

    if parsed_version >= version.parse('1.40.0'):
        return
    else:
        raise ValueError(
            'Incompatiable serverless version, please install version 1.40.0 or greater')


def generate_serverless_bundle(bento_service, platform, archive_path, extra_args):
    check_serverless_compatiable_version()

    provider = SERVERLESS_PROVIDER[platform]
    TEMP_FOLDER_PATH = './temp_bento_serverless_' + datetime.now().isoformat()
    parsed_extra_args = default_serverless_parser.parse_args(extra_args)

    # Because Serverless framework will modify even absolute path with CWD.
    # So, if user provide an absolute path as parameter, we will generate the serverless
    # in the current directory and after our modification we will copy to user's desired
    # path and delete the temporary one we created.
    output_path = TEMP_FOLDER_PATH

    serverless_config_file = os.path.join(output_path, 'serverless.yml')
    generate_base_serverless_files(output_path, provider, bento_service.name)

    if provider != 'google-python':
        update_serverless_configuration_for_aws(bento_service, serverless_config_file,
                                                parsed_extra_args)
    else:
        update_serverless_configuration_for_google(bento_service, serverless_config_file,
                                                   parsed_extra_args)

    generate_handler_py(bento_service, output_path, provider)

    shutil.copy(os.path.join(archive_path, 'requirements.txt'), output_path)
    add_model_service_archive(bento_service, archive_path, output_path)

    return os.path.realpath(TEMP_FOLDER_PATH)


def deploy_with_serverless(bento_service, platform, archive_path, extra_args):
    output_path = generate_serverless_bundle(bento_service, platform, archive_path, extra_args)
    deploy_serverless_file(output_path)
    return output_path
