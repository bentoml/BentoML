import os
from ruamel.yaml import YAML

from bentoml.deployment.serverless.aws_lambda import (
    generate_aws_lambda_serverless_config,
    generate_aws_lambda_handler_py)


def test_generate_aws_lambda_serverless_config(tmpdir):
    python_version = '3.7.2'
    deployment_name = 'test_deployment_lambda'
    api_names = ['predict']
    region = 'us-west-test'
    namespace = 'namespace'
    generate_aws_lambda_serverless_config(
        python_version, deployment_name, api_names, tmpdir, region, namespace
    )
    config_path = os.path.join(tmpdir, 'serverless.yml')
    yaml = YAML()
    with open(config_path, 'rb') as f:
        yaml_data = yaml.load(f.read())
    assert yaml_data['service'] == deployment_name
    assert yaml_data['functions']['predict']['handler'] == 'handler.predict'
    assert yaml_data['provider']['region'] == 'us-west-test'


def test_generate_aws_lambda_handler_py(tmpdir):
    bento_name = 'bento_name'
    api_names = ['predict', 'second_predict']
    generate_aws_lambda_handler_py(bento_name, api_names, tmpdir)
    handler_path = os.path.join(tmpdir, 'handler.py')
    with open(handler_path, 'r') as f:
        handler_content = f.read()

    assert 'from bento_name import load' in handler_content
    assert 'def predict(event, context):' in handler_content
    assert 'def second_predict(event, context):' in handler_content

