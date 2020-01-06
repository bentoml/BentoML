from mock import MagicMock, patch

from bentoml.yatai.yatai_service_impl import YataiService


@patch('bentoml.yatai.yatai_service_impl.BentoRepository', MagicMock())
@patch('bentoml.yatai.yatai_service_impl.init_db', MagicMock())
@patch('bentoml.yatai.yatai_service_impl.DeploymentStore', MagicMock())
@patch('bentoml.yatai.yatai_service_impl.BentoMetadataStore', MagicMock())
@patch(
    'boto3.session.Session',
    MagicMock(return_value=MagicMock(region_name='mock_region')),
)
def test_create_yatai_server():
    mock_db_url = 'mock_url'
    mock_repo_base_url = 'mock_repo_base_url'
    mock_default_namespace = 'mock_namespace'
    service = YataiService(
        db_url=mock_db_url,
        repo_base_url=mock_repo_base_url,
        default_namespace=mock_default_namespace,
    )
    assert service.default_namespace == 'mock_namespace'
    assert service.default_aws_region == 'mock_region'
