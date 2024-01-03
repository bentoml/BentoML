from __future__ import annotations

import typing as t
from datetime import datetime
from unittest.mock import patch

import attr
import pytest

from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.schemas.modelschemas import AccessControl
from bentoml._internal.cloud.schemas.modelschemas import DeploymentServiceConfig
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetHPAConf
from bentoml._internal.cloud.schemas.schemasv1 import BentoFullSchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoImageBuildStatus
from bentoml._internal.cloud.schemas.schemasv1 import BentoManifestSchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoRepositorySchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoUploadStatus
from bentoml._internal.cloud.schemas.schemasv1 import ClusterListSchema
from bentoml._internal.cloud.schemas.schemasv1 import ClusterSchema
from bentoml._internal.cloud.schemas.schemasv1 import DeploymentRevisionStatus
from bentoml._internal.cloud.schemas.schemasv1 import DeploymentStatus
from bentoml._internal.cloud.schemas.schemasv1 import LabelItemSchema
from bentoml._internal.cloud.schemas.schemasv1 import ResourceType
from bentoml._internal.cloud.schemas.schemasv1 import UserSchema
from bentoml._internal.cloud.schemas.schemasv2 import (
    CreateDeploymentSchema as CreateDeploymentSchemaV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    DeploymentFullSchema as DeploymentFullSchemaV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    DeploymentRevisionSchema as DeploymentRevisionSchemaV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    DeploymentTargetConfig as DeploymentTargetConfigV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    DeploymentTargetSchema as DeploymentTargetSchemaV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    UpdateDeploymentSchema as UpdateDeploymentSchemaV2,
)

if t.TYPE_CHECKING:
    from unittest.mock import MagicMock


@attr.define
class DummyUpdateSchema(UpdateDeploymentSchemaV2):
    urls: t.List[str] = attr.Factory(
        list
    )  # place holder for urls that's assigned to deployment._urls


@pytest.fixture(name="rest_client", scope="function")
def fixture_rest_client() -> RestApiClient:
    def dummy_create_deployment(
        create_schema: CreateDeploymentSchemaV2, cluster_name: str
    ):
        return create_schema

    def dummy_update_deployment(
        update_schema: UpdateDeploymentSchemaV2, cluster_name: str, deployment_name: str
    ):
        from bentoml._internal.utils import bentoml_cattr

        return bentoml_cattr.structure(attr.asdict(update_schema), DummyUpdateSchema)

    def dummy_get_deployment(cluster_name: str, deployment_name: str):
        if deployment_name == "test-distributed":
            return DeploymentFullSchemaV2(
                distributed=True,
                latest_revision=DeploymentRevisionSchemaV2(
                    targets=[
                        DeploymentTargetSchemaV2(
                            bento=BentoFullSchema(
                                uid="",
                                created_at=datetime(2023, 5, 25),
                                updated_at=None,
                                deleted_at=None,
                                name="123",
                                resource_type=ResourceType.BENTO,
                                labels=[],
                                description="",
                                version="",
                                image_build_status=BentoImageBuildStatus.PENDING,
                                upload_status=BentoUploadStatus.SUCCESS,
                                upload_finished_reason="",
                                presigned_upload_url="",
                                presigned_download_url="",
                                manifest=BentoManifestSchema(
                                    service="",
                                    bentoml_version="",
                                    size_bytes=0,
                                    apis={},
                                    models=["iris_clf:ddaex6h2vw6kwcvj"],
                                ),
                                build_at=datetime(2023, 5, 25),
                                repository=BentoRepositorySchema(
                                    uid="",
                                    created_at="",
                                    updated_at=None,
                                    deleted_at=None,
                                    name="abc",
                                    resource_type=ResourceType.BENTO_REPOSITORY,
                                    labels=[],
                                    description="",
                                    latest_bento="",
                                ),
                            ),
                            config=DeploymentTargetConfigV2(
                                access_type=AccessControl.PUBLIC,
                                envs=[
                                    LabelItemSchema(key="env_key", value="env_value")
                                ],
                                services={
                                    "irisclassifier": DeploymentServiceConfig(
                                        instance_type="t3-small",
                                        scaling=DeploymentTargetHPAConf(
                                            min_replicas=1, max_replicas=1
                                        ),
                                        deployment_strategy=DeploymentStrategy.RollingUpdate,
                                    ),
                                    "preprocessing": DeploymentServiceConfig(
                                        instance_type="t3-small",
                                        scaling=DeploymentTargetHPAConf(
                                            min_replicas=1, max_replicas=1
                                        ),
                                        deployment_strategy=DeploymentStrategy.RollingUpdate,
                                    ),
                                },
                            ),
                            uid="",
                            created_at=datetime(2023, 5, 1),
                            updated_at=None,
                            deleted_at=None,
                            name="",
                            resource_type=ResourceType.DEPLOYMENT_REVISION,
                            labels=[],
                            creator=user,
                        )
                    ],
                    uid="",
                    created_at=datetime(2023, 5, 1),
                    updated_at=None,
                    deleted_at=None,
                    name="test=xxx",
                    resource_type=ResourceType.DEPLOYMENT_REVISION,
                    labels=[],
                    creator=user,
                    status=DeploymentRevisionStatus.ACTIVE,
                ),
                uid="",
                created_at=datetime(2023, 5, 1),
                updated_at=None,
                deleted_at=None,
                name="test=xxx",
                resource_type=ResourceType.DEPLOYMENT_REVISION,
                labels=[],
                creator=user,
                status=DeploymentStatus.Running,
                cluster=ClusterSchema(
                    uid="",
                    name="default",
                    resource_type=ResourceType.CLUSTER,
                    labels=[],
                    description="",
                    creator=user,
                    created_at=datetime(2023, 5, 1),
                    updated_at=None,
                    deleted_at=None,
                ),
                kube_namespace="",
            )

        else:
            return DeploymentFullSchemaV2(
                distributed=False,
                latest_revision=DeploymentRevisionSchemaV2(
                    targets=[
                        DeploymentTargetSchemaV2(
                            bento=BentoFullSchema(
                                uid="",
                                created_at=datetime(2023, 5, 25),
                                updated_at=None,
                                deleted_at=None,
                                name="123",
                                resource_type=ResourceType.BENTO,
                                labels=[],
                                description="",
                                version="",
                                image_build_status=BentoImageBuildStatus.PENDING,
                                upload_status=BentoUploadStatus.SUCCESS,
                                upload_finished_reason="",
                                presigned_upload_url="",
                                presigned_download_url="",
                                manifest=BentoManifestSchema(
                                    service="",
                                    bentoml_version="",
                                    size_bytes=0,
                                    apis={},
                                    models=["iris_clf:ddaex6h2vw6kwcvj"],
                                ),
                                build_at=datetime(2023, 5, 25),
                                repository=BentoRepositorySchema(
                                    uid="",
                                    created_at="",
                                    updated_at=None,
                                    deleted_at=None,
                                    name="abc",
                                    resource_type=ResourceType.BENTO_REPOSITORY,
                                    labels=[],
                                    description="",
                                    latest_bento="",
                                ),
                            ),
                            config=DeploymentTargetConfigV2(
                                access_type=AccessControl.PUBLIC,
                                scaling=DeploymentTargetHPAConf(
                                    min_replicas=3, max_replicas=5
                                ),
                                deployment_strategy=DeploymentStrategy.RollingUpdate,
                                envs=[
                                    LabelItemSchema(key="env_key", value="env_value")
                                ],
                            ),
                            uid="",
                            created_at=datetime(2023, 5, 1),
                            updated_at=None,
                            deleted_at=None,
                            name="",
                            resource_type=ResourceType.DEPLOYMENT_REVISION,
                            labels=[],
                            creator=user,
                        )
                    ],
                    uid="",
                    created_at=datetime(2023, 5, 1),
                    updated_at=None,
                    deleted_at=None,
                    name="test=xxx",
                    resource_type=ResourceType.DEPLOYMENT_REVISION,
                    labels=[],
                    creator=user,
                    status=DeploymentRevisionStatus.ACTIVE,
                ),
                uid="",
                created_at=datetime(2023, 5, 1),
                updated_at=None,
                deleted_at=None,
                name="test=xxx",
                resource_type=ResourceType.DEPLOYMENT_REVISION,
                labels=[],
                creator=user,
                status=DeploymentStatus.Running,
                cluster=ClusterSchema(
                    uid="",
                    name="default",
                    resource_type=ResourceType.CLUSTER,
                    labels=[],
                    description="",
                    creator=user,
                    created_at=datetime(2023, 5, 1),
                    updated_at=None,
                    deleted_at=None,
                ),
                kube_namespace="",
            )

    client = RestApiClient("", "")
    user = UserSchema(name="", email="", first_name="", last_name="")
    client.v2.create_deployment = dummy_create_deployment  # type: ignore
    client.v2.update_deployment = dummy_update_deployment  # type: ignore
    client.v1.get_cluster_list = lambda params: ClusterListSchema(
        start=0,
        count=0,
        total=0,
        items=[
            ClusterSchema(
                uid="",
                name="default",
                resource_type=ResourceType.CLUSTER,
                labels=[],
                description="",
                creator=user,
                created_at=datetime(2023, 5, 1),
                updated_at=None,
                deleted_at=None,
            )
        ],
    )  # type: ignore

    client.v2.get_deployment = dummy_get_deployment

    return client


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment(mock_get_client: MagicMock, rest_client: RestApiClient):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(bento="abc:123")
    # assert expected schema
    assert deployment._schema == CreateDeploymentSchemaV2(
        scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=1),
        bento="abc:123",
        name="",
        cluster="default",
        access_type=AccessControl.PUBLIC,
        distributed=False,
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_custom_standalone(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(
        bento="abc:123",
        name="custom-name",
        scaling_min=2,
        scaling_max=4,
        access_type="private",
        cluster_name="custom-cluster",
        envs=[{"key": "env_key", "value": "env_value"}],
        strategy="RollingUpdate",
    )
    # assert expected schema
    assert deployment._schema == CreateDeploymentSchemaV2(
        bento="abc:123",
        name="custom-name",
        cluster="custom-cluster",
        access_type=AccessControl.PRIVATE,
        scaling=DeploymentTargetHPAConf(min_replicas=2, max_replicas=4),
        distributed=False,
        deployment_strategy=DeploymentStrategy.RollingUpdate,
        envs=[LabelItemSchema(key="env_key", value="env_value")],
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_scailing_only_min(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(bento="abc:123", scaling_min=3)
    # assert expected schema
    assert deployment._schema == CreateDeploymentSchemaV2(
        bento="abc:123",
        name="",
        cluster="default",
        access_type=AccessControl.PUBLIC,
        scaling=DeploymentTargetHPAConf(min_replicas=3, max_replicas=3),
        distributed=False,
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_scailing_only_max(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(bento="abc:123", scaling_max=3)
    # assert expected schema
    assert deployment._schema == CreateDeploymentSchemaV2(
        bento="abc:123",
        name="",
        cluster="default",
        access_type=AccessControl.PUBLIC,
        scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=3),
        distributed=False,
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_scailing_mismatch_min_max(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(bento="abc:123", scaling_min=3, scaling_max=2)
    # assert expected schema
    assert deployment._schema == CreateDeploymentSchemaV2(
        bento="abc:123",
        name="",
        cluster="default",
        access_type=AccessControl.PUBLIC,
        scaling=DeploymentTargetHPAConf(min_replicas=2, max_replicas=2),
        distributed=False,
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_config_dct(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    config_dct = {
        "services": {
            "irisclassifier": {"scaling": {"max_replicas": 2, "min_replicas": 1}},
            "preprocessing": {"scaling": {"max_replicas": 2}},
        },
        "envs": [{"key": "env_key", "value": "env_value"}],
        "bentoml_config_overrides": {
            "irisclassifier": {
                "resources": {
                    "cpu": "300m",
                    "memory": "500m",
                },
            }
        },
    }
    deployment = Deployment.create(bento="abc:123", config_dct=config_dct)
    # assert expected schema
    assert deployment._schema == CreateDeploymentSchemaV2(
        bento="abc:123",
        name="",
        cluster="default",
        access_type=AccessControl.PUBLIC,
        distributed=True,
        services={
            "irisclassifier": DeploymentServiceConfig(
                scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=2)
            ),
            "preprocessing": DeploymentServiceConfig(
                scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=2)
            ),
        },
        envs=[LabelItemSchema(key="env_key", value="env_value")],
        bentoml_config_overrides={
            "irisclassifier": {
                "resources": {
                    "cpu": "300m",
                    "memory": "500m",
                },
            }
        },
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment(mock_get_client: MagicMock, rest_client: RestApiClient):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(
        name="test",
        bento="abc:1234",
        access_type="private",
        envs=[{"key": "env_key2", "value": "env_value2"}],
        strategy="Recreate",
    )
    # assert expected schema
    assert deployment._schema == DummyUpdateSchema(
        bento="abc:1234",
        access_type=AccessControl.PRIVATE,
        scaling=DeploymentTargetHPAConf(min_replicas=3, max_replicas=5),
        deployment_strategy=DeploymentStrategy.Recreate,
        envs=[LabelItemSchema(key="env_key2", value="env_value2")],
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_scaling_only_min(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(name="test", scaling_min=1)
    # assert expected schema
    assert deployment._schema == DummyUpdateSchema(
        bento="abc:123",
        access_type=AccessControl.PUBLIC,
        scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=5),
        deployment_strategy=DeploymentStrategy.RollingUpdate,
        envs=[LabelItemSchema(key="env_key", value="env_value")],
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_scaling_only_max(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(name="test", scaling_max=3)
    # assert expected schema
    assert deployment._schema == DummyUpdateSchema(
        bento="abc:123",
        access_type=AccessControl.PUBLIC,
        scaling=DeploymentTargetHPAConf(min_replicas=3, max_replicas=3),
        deployment_strategy=DeploymentStrategy.RollingUpdate,
        envs=[LabelItemSchema(key="env_key", value="env_value")],
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_scaling_too_big_min(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(name="test", scaling_min=10)
    # assert expected schema
    assert deployment._schema == DummyUpdateSchema(
        bento="abc:123",
        access_type=AccessControl.PUBLIC,
        scaling=DeploymentTargetHPAConf(min_replicas=5, max_replicas=5),
        deployment_strategy=DeploymentStrategy.RollingUpdate,
        envs=[LabelItemSchema(key="env_key", value="env_value")],
    )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_distributed(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    config_dct = {
        "services": {
            "irisclassifier": {"scaling": {"max_replicas": 50}},
            "preprocessing": {"instance_type": "t3-large"},
        }
    }
    deployment = Deployment.update(name="test-distributed", config_dct=config_dct)
    # assert expected schema
    assert deployment._schema == DummyUpdateSchema(
        bento="abc:123",
        access_type=AccessControl.PUBLIC,
        envs=[LabelItemSchema(key="env_key", value="env_value")],
        services={
            "irisclassifier": DeploymentServiceConfig(
                instance_type="t3-small",
                scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=50),
                deployment_strategy=DeploymentStrategy.RollingUpdate,
            ),
            "preprocessing": DeploymentServiceConfig(
                instance_type="t3-large",
                scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=1),
                deployment_strategy=DeploymentStrategy.RollingUpdate,
            ),
        },
    )
