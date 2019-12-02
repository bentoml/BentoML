import os

import boto3
from mock import MagicMock, patch, Mock
from moto import mock_s3
from ruamel.yaml import YAML

from bentoml.deployment.aws_lambda import (
    AwsLambdaDeploymentOperator,
    _create_aws_lambda_cloudformation_template_file,
    init_sam_project,
)
from bentoml.proto.deployment_pb2 import Deployment, DeploymentState
from bentoml.proto.repository_pb2 import (
    Bento,
    BentoUri,
    BentoServiceMetadata,
    GetBentoResponse,
)
from bentoml.proto import status_pb2

mock_s3_bucket_name = 'test_deployment_bucket'
mock_s3_prefix = 'prefix'
mock_s3_path = 's3://{}/{}'.format(mock_s3_bucket_name, mock_s3_prefix)


def create_yatai_service_mock(repo_storage_type=BentoUri.LOCAL):
    bento_pb = Bento(name='bento_test_name', version='version1.1.1')
    if repo_storage_type == BentoUri.LOCAL:
        bento_pb.uri.uri = '/tmp/path/to/bundle'
    bento_pb.uri.type = repo_storage_type
    api = BentoServiceMetadata.BentoServiceApi(name='predict')
    bento_pb.bento_service_metadata.apis.extend([api])
    bento_pb.bento_service_metadata.env.python_version = '3.7.0'
    get_bento_response = GetBentoResponse(bento=bento_pb)

    yatai_service_mock = MagicMock()
    yatai_service_mock.GetBento.return_value = get_bento_response
    return yatai_service_mock


def generate_lambda_deployment_pb():
    test_deployment_pb = Deployment(name='test_aws_lambda', namespace='test-namespace')
    test_deployment_pb.spec.bento_name = 'bento_name'
    test_deployment_pb.spec.bento_version = 'v1.0.0'
    # DeploymentSpec.DeploymentOperator.AWS_LAMBDA
    test_deployment_pb.spec.operator = 3
    test_deployment_pb.spec.aws_lambda_operator_config.region = 'us-west-2'
    test_deployment_pb.spec.aws_lambda_operator_config.api_name = 'predict'
    test_deployment_pb.spec.aws_lambda_operator_config.memory_size = 3008
    test_deployment_pb.spec.aws_lambda_operator_config.timeout = 6

    return test_deployment_pb


def test_aws_lambda_app_py(monkeypatch):
    def test_predict(value):
        return value

    class Mock_bento_service(object):
        def _load_artifacts(self, path):
            return

        def get_service_api(self, name):
            if name == 'predict':
                mock_api = Mock()
                mock_api.handle_aws_lambda_event = test_predict
                return mock_api

        _artifacts = []

    monkeypatch.setenv('BENTOML_BENTO_SERVICE_NAME', 'Mock_bento_service')
    monkeypatch.setenv('BENTOML_S3_BUCKET', 'Mock_s3_bucket')
    monkeypatch.setenv('BENTOML_DEPLOYMENT_PATH_PREFIX', 'deployment/prefix')
    monkeypatch.setenv('BENTOML_ARTIFACTS_PREFIX', 'mock_artifacts_prefix')
    monkeypatch.setenv('BENTOML_API_NAME', 'predict')

    def mock_lambda_app(func):
        @mock_s3
        @patch('bentoml.utils.s3.download_directory_from_s3', return_value=None)
        def mock_wrapper(*args, **kwargs):
            conn = boto3.client('s3', region_name='us-west-2')
            conn.create_bucket(Bucket=mock_s3_bucket_name)
            mock_artifact_key = 'mock_artifact_prefix/model.pkl'
            conn.put_object(
                Bucket=mock_s3_bucket_name, Key=mock_artifact_key, Body='mock_body'
            )
            return func(*args, **kwargs)

        return mock_wrapper

    @mock_lambda_app
    @patch('bentoml.bundler.load_bento_service_class', return_value=Mock_bento_service)
    def return_predict_func(mock_load_service, mock_download_artifacts):
        from bentoml.deployment.aws_lambda.lambda_app import predict

        return predict

    predict = return_predict_func()
    assert predict(1, None) == 1


@patch('shutil.rmtree', MagicMock())
@patch('shutil.copytree', MagicMock())
@patch('bentoml.deployment.aws_lambda.utils.cleanup_build_files', MagicMock())
@patch('bentoml.deployment.aws_lambda.utils.call_sam_command', autospec=True)
def test_init_sam_project(mock_call_sam, tmpdir):
    mock_sam_project_path = os.path.join(tmpdir, 'mock_sam_project')
    mock_bento_bundle_path = os.path.join(tmpdir, 'mock_bento_service')
    mock_deployment_name = 'mock_deployment'
    mock_bento_name = 'mock_bento_name'
    mock_api_names = ['predict']
    mock_region = 'us-west-2'
    mock_call_sam.return_value = 0, 'stdout', 'stderr'
    os.mkdir(mock_sam_project_path)
    os.mkdir(mock_bento_bundle_path)
    open(os.path.join(mock_bento_bundle_path, 'requirements.txt'), 'w').close()

    init_sam_project(
        mock_sam_project_path,
        mock_bento_bundle_path,
        mock_deployment_name,
        mock_bento_name,
        mock_api_names,
        mock_region,
    )
    assert os.path.isfile(
        os.path.join(mock_sam_project_path, mock_deployment_name, 'app.py')
    )
    assert os.path.isfile(
        os.path.join(mock_sam_project_path, mock_deployment_name, 'requirements.txt')
    )
    assert os.path.isfile(
        os.path.join(mock_sam_project_path, mock_deployment_name, '__init__.py')
    )


def test_generate_aws_lambda_template_yaml(tmpdir):
    deployment_name = 'deployment_name'
    api_names = ['predict', 'classify']
    s3_bucket_name = 'test_bucket'
    py_runtime = 'python3.7'
    memory_size = 3008
    timeout = 6
    _create_aws_lambda_cloudformation_template_file(
        str(tmpdir),
        deployment_name=deployment_name,
        deployment_path_prefix='mock/deployment/path/prefix',
        api_names=api_names,
        bento_service_name='mock_bento_service_name',
        artifacts_prefix='mock_artifacts_prefix',
        s3_bucket_name=s3_bucket_name,
        py_runtime=py_runtime,
        memory_size=memory_size,
        timeout=timeout,
    )
    template_path = os.path.join(str(tmpdir), 'template.yaml')
    yaml = YAML()
    with open(template_path, 'rb') as f:
        yaml_data = yaml.load(f.read())
    assert yaml_data['Resources']['predict']['Properties']['Runtime'] == py_runtime
    assert yaml_data['Resources']['classify']['Properties']['Handler'] == 'app.classify'


def mock_lambda_related_operations(func):
    @patch('subprocess.check_output', MagicMock())
    @mock_s3
    def mock_wrapper(*args, **kwargs):
        conn = boto3.client('s3', region_name='us-west-2')
        conn.create_bucket(Bucket=mock_s3_bucket_name)
        return func(*args, **kwargs)

    return mock_wrapper()


@mock_lambda_related_operations
@patch('shutil.rmtree', MagicMock())
@patch('shutil.copytree', MagicMock())
@patch('shutil.copy', MagicMock())
@patch('os.listdir', MagicMock())
@patch('bentoml.deployment.aws_lambda.init_sam_project', MagicMock())
@patch('bentoml.deployment.aws_lambda.lambda_package', MagicMock())
@patch(
    'bentoml.deployment.aws_lambda.validate_lambda_template',
    MagicMock(return_value=None),
)
@patch('bentoml.deployment.aws_lambda.lambda_deploy', MagicMock(return_value=None))
def test_aws_lambda_apply_success():
    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_lambda_deployment_pb()
    deployment_operator = AwsLambdaDeploymentOperator()

    result_pb = deployment_operator.apply(test_deployment_pb, yatai_service_mock, None)

    assert result_pb.status.status_code == status_pb2.Status.OK
    assert result_pb.deployment.state.state == DeploymentState.PENDING


def test_aws_lambda_describe_still_in_progress():
    def mock_cf_response(self, op_name, kwarg):
        if op_name == 'DescribeStacks':
            return {'Stacks': [{'StackStatus': 'CREATE_IN_PROGRESS'}]}
        else:
            raise Exception('This test does not handle operation {}'.format(op_name))

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_lambda_deployment_pb()
    with patch('botocore.client.BaseClient._make_api_call', new=mock_cf_response):
        deployment_operator = AwsLambdaDeploymentOperator()
        result_pb = deployment_operator.describe(test_deployment_pb, yatai_service_mock)
        assert result_pb.status.status_code == status_pb2.Status.OK
        assert result_pb.state.state == DeploymentState.PENDING


def test_aws_lambda_describe_success():
    def mock_cf_response(self, op_name, kwarg):
        if op_name == 'DescribeStacks':
            return {
                'Stacks': [
                    {
                        'StackStatus': 'CREATE_COMPLETE',
                        'Outputs': [
                            {
                                'OutputKey': 'EndpointUrl',
                                'OutputValue': 'https://somefakelink.amazonaws.com'
                                '/prod/predict',
                            }
                        ],
                    }
                ]
            }
        else:
            raise Exception('This test does not handle operation {}'.format(op_name))

    yatai_service_mock = create_yatai_service_mock()
    test_deployment_pb = generate_lambda_deployment_pb()
    with patch('botocore.client.BaseClient._make_api_call', new=mock_cf_response):
        deployment_operator = AwsLambdaDeploymentOperator()
        result_pb = deployment_operator.describe(test_deployment_pb, yatai_service_mock)
        assert result_pb.status.status_code == status_pb2.Status.OK
        assert result_pb.state.state == DeploymentState.RUNNING
