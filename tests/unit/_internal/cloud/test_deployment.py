from __future__ import annotations

import typing as t
from datetime import datetime
from unittest.mock import patch

import attr
import pytest

from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.cloud.schemas.modelschemas import AccessControl
from bentoml._internal.cloud.schemas.modelschemas import DeploymentServiceConfig
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy
from bentoml._internal.cloud.schemas.modelschemas import DeploymentTargetHPAConf
from bentoml._internal.cloud.schemas.modelschemas import EnvItemSchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoFullSchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoImageBuildStatus
from bentoml._internal.cloud.schemas.schemasv1 import BentoManifestSchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoRepositorySchema
from bentoml._internal.cloud.schemas.schemasv1 import BentoUploadStatus
from bentoml._internal.cloud.schemas.schemasv1 import ClusterListSchema
from bentoml._internal.cloud.schemas.schemasv1 import ClusterSchema
from bentoml._internal.cloud.schemas.schemasv1 import DeploymentRevisionStatus
from bentoml._internal.cloud.schemas.schemasv1 import DeploymentStatus
from bentoml._internal.cloud.schemas.schemasv1 import ResourceType
from bentoml._internal.cloud.schemas.schemasv1 import UserSchema
from bentoml._internal.cloud.schemas.schemasv2 import (
    CreateDeploymentSchema as CreateDeploymentSchemaV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    DeploymentConfigSchema as DeploymentConfigSchemaV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    DeploymentFullSchema as DeploymentFullSchemaV2,
)
from bentoml._internal.cloud.schemas.schemasv2 import (
    DeploymentRevisionSchema as DeploymentRevisionSchemaV2,
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


def dummy_generate_deployment_schema(
    name: str,
    cluster: str | None,
    update_schema: UpdateDeploymentSchemaV2,
):
    user = UserSchema(name="", email="", first_name="", last_name="")
    bento = update_schema.bento.split(":")
    if len(bento) == 2:
        bento_name = bento[0]
        bento_version = bento[1]
    else:
        bento_name = bento[0]
        bento_version = ""
    dummy_bento = BentoFullSchema(
        uid="",
        created_at=datetime(2023, 5, 25),
        updated_at=None,
        deleted_at=None,
        name=bento_name,
        resource_type=ResourceType.BENTO,
        labels=[],
        description="",
        version=bento_version,
        image_build_status=BentoImageBuildStatus.PENDING,
        upload_status=BentoUploadStatus.SUCCESS,
        upload_finished_reason="",
        presigned_upload_url="",
        presigned_download_url="",
        manifest=BentoManifestSchema(
            service="",
            entry_service="",
            bentoml_version="",
            size_bytes=0,
            apis={},
            models=["iris_clf:ddaex6h2vw6kwcvj"],
        ),
        build_at=datetime(2023, 5, 25),
        repository=BentoRepositorySchema(
            uid="",
            created_at=datetime(2023, 5, 1),
            updated_at=None,
            deleted_at=None,
            name=bento_name,
            resource_type=ResourceType.BENTO_REPOSITORY,
            labels=[],
            description="",
            latest_bento=None,
        ),
    )
    return DeploymentFullSchemaV2(
        latest_revision=DeploymentRevisionSchemaV2(
            targets=[
                DeploymentTargetSchemaV2(
                    bento=dummy_bento,
                    config=DeploymentConfigSchemaV2(
                        access_type=update_schema.access_type,
                        envs=update_schema.envs,
                        services=update_schema.services,
                    ),
                    uid="",
                    created_at=datetime(2023, 5, 1),
                    updated_at=None,
                    deleted_at=None,
                    name=name,
                    resource_type=ResourceType.DEPLOYMENT_REVISION,
                    labels=[],
                    creator=user,
                )
            ],
            uid="",
            created_at=datetime(2023, 5, 1),
            updated_at=None,
            deleted_at=None,
            name=name,
            resource_type=ResourceType.DEPLOYMENT_REVISION,
            labels=[],
            creator=user,
            status=DeploymentRevisionStatus.ACTIVE,
        ),
        uid="",
        created_at=datetime(2023, 5, 1),
        updated_at=None,
        deleted_at=None,
        name=name,
        resource_type=ResourceType.DEPLOYMENT_REVISION,
        labels=[],
        creator=user,
        status=DeploymentStatus.Running,
        cluster=ClusterSchema(
            uid="",
            name="default",
            host_cluster_display_name=cluster,
            organization_name="default_organization",
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


@pytest.fixture(name="rest_client", scope="function")
def fixture_rest_client() -> RestApiClient:
    def dummy_create_deployment(
        create_schema: CreateDeploymentSchemaV2, cluster: str | None = None
    ):
        if create_schema.name is None:
            create_schema.name = "empty_name"
        if cluster is None:
            cluster = "default_display_name"
        return dummy_generate_deployment_schema(
            create_schema.name, cluster, create_schema
        )

    def dummy_update_deployment(
        name: str,
        update_schema: UpdateDeploymentSchemaV2,
        cluster: str | None = None,
    ):
        if cluster is None:
            cluster = "default_display_name"
        return dummy_generate_deployment_schema(name, cluster, update_schema)

    def dummy_get_deployment(
        name: str,
        cluster: str | None = None,
    ):
        if cluster is None:
            cluster = "default_display_name"
        if name == "test-distributed":
            return dummy_generate_deployment_schema(
                name,
                cluster,
                UpdateDeploymentSchemaV2(
                    bento="abc:123",
                    envs=[EnvItemSchema(name="env_key", value="env_value")],
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
            )
        else:
            return dummy_generate_deployment_schema(
                name,
                cluster,
                UpdateDeploymentSchemaV2(
                    bento="abc:123",
                    access_type=AccessControl.PUBLIC,
                    services={
                        "irisclassifier": DeploymentServiceConfig(
                            scaling=DeploymentTargetHPAConf(
                                min_replicas=3, max_replicas=5
                            ),
                            deployment_strategy=DeploymentStrategy.RollingUpdate,
                        )
                    },
                    envs=[EnvItemSchema(name="env_key", value="env_value")],
                ),
            )

    client = RestApiClient("", "")
    user = UserSchema(name="", email="", first_name="", last_name="")
    client.v2.create_deployment = dummy_create_deployment  # type: ignore
    client.v2.update_deployment = dummy_update_deployment  # type: ignore
    client.v2.get_deployment = dummy_get_deployment  # type: ignore
    client.v1.get_cluster_list = lambda params: ClusterListSchema(
        start=0,
        count=0,
        total=0,
        items=[
            ClusterSchema(
                uid="",
                name="default",
                host_cluster_display_name="default_display_name",
                organization_name="default_organization",
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

    return client


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment(mock_get_client: MagicMock, rest_client: RestApiClient):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "bento": "abc:123",
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.name == "empty_name"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PUBLIC
    for service in config.services.values():
        assert service.scaling == DeploymentTargetHPAConf(
            min_replicas=1, max_replicas=1
        )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_custom_standalone(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "bento": "abc:123",
                "name": "custom-name",
                "access_type": "private",
                "cluster": "custom-cluster",
                "envs": [{"name": "env_key", "value": "env_value"}],
                "services": {
                    "irisclassifier": {
                        "deployment_strategy": "RollingUpdate",
                        "scaling": {"min_replicas": 2, "max_replicas": 4},
                    },
                },
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "custom-cluster"
    assert deployment.name == "custom-name"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PRIVATE
    assert config.envs == [EnvItemSchema(name="env_key", value="env_value")]
    for service in config.services.values():
        assert service.scaling == DeploymentTargetHPAConf(
            min_replicas=2, max_replicas=4
        )
        assert service.deployment_strategy == DeploymentStrategy.RollingUpdate


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_scailing_only_min(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "bento": "abc:123",
                "services": {
                    "irisclassifier": {
                        "scaling": {"min_replicas": 3},
                    },
                },
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.name == "empty_name"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PUBLIC
    for service in config.services.values():
        assert service.scaling == DeploymentTargetHPAConf(
            min_replicas=3, max_replicas=3
        )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_scailing_only_max(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "bento": "abc:123",
                "services": {
                    "irisclassifier": {
                        "scaling": {"max_replicas": 3},
                    },
                },
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.name == "empty_name"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PUBLIC
    for service in config.services.values():
        assert service.scaling == DeploymentTargetHPAConf(
            min_replicas=1, max_replicas=3
        )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_create_deployment_config_dict(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.create(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "bento": "abc:123",
                "services": {
                    "irisclassifier": {
                        "scaling": {"max_replicas": 2, "min_replicas": 1},
                        "config_overrides": {
                            "resources": {"cpu": "300m", "memory": "500m"}
                        },
                    },
                    "preprocessing": {"scaling": {"max_replicas": 2}},
                },
                "envs": [{"name": "env_key", "value": "env_value"}],
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.name == "empty_name"
    config = deployment.get_config(refetch=False)
    assert config.services == {
        "irisclassifier": DeploymentServiceConfig(
            scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=2),
            config_overrides={
                "resources": {
                    "cpu": "300m",
                    "memory": "500m",
                },
            },
        ),
        "preprocessing": DeploymentServiceConfig(
            scaling=DeploymentTargetHPAConf(min_replicas=1, max_replicas=2)
        ),
    }


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment(mock_get_client: MagicMock, rest_client: RestApiClient):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "name": "test",
                "bento": "abc:1234",
                "access_type": "private",
                "envs": [{"name": "env_key2", "value": "env_value2"}],
                "services": {
                    "irisclassifier": {
                        "deployment_strategy": "Recreate",
                    },
                },
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.get_bento(refetch=False) == "abc:1234"
    assert deployment.name == "test"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PRIVATE
    assert config.envs == [EnvItemSchema(name="env_key2", value="env_value2")]
    for service in config.services.values():
        assert service.scaling == DeploymentTargetHPAConf(
            min_replicas=3, max_replicas=5
        )
        assert service.deployment_strategy == DeploymentStrategy.Recreate


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_scaling_only_min(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "name": "test",
                "services": {
                    "irisclassifier": {
                        "scaling": {"min_replicas": 1},
                    },
                },
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.name == "test"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PUBLIC
    assert config.envs == [EnvItemSchema(name="env_key", value="env_value")]
    for service in config.services.values():
        assert service.scaling == DeploymentTargetHPAConf(
            min_replicas=1, max_replicas=5
        )
        assert service.deployment_strategy == DeploymentStrategy.RollingUpdate


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_scaling_only_max(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "name": "test",
                "services": {
                    "irisclassifier": {
                        "scaling": {"max_replicas": 3},
                    },
                },
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.name == "test"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PUBLIC
    assert config.envs == [EnvItemSchema(name="env_key", value="env_value")]
    for service in config.services.values():
        assert service.scaling == DeploymentTargetHPAConf(
            min_replicas=3, max_replicas=3
        )
        assert service.deployment_strategy == DeploymentStrategy.RollingUpdate


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_scaling_too_big_min(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    try:
        Deployment.update(
            deployment_config_params=DeploymentConfigParameters(
                cfg_dict={
                    "name": "test",
                    "services": {
                        "irisclassifier": {"scaling": {"min_replicas": 10}},
                    },
                },
                service_name="irisclassifier",
            )
        )
    except Exception as e:
        assert (
            "min scaling values must be less than or equal to max scaling values"
            in str(e)
        )


@patch("bentoml._internal.cloud.deployment.get_rest_api_client")
def test_update_deployment_distributed(
    mock_get_client: MagicMock, rest_client: RestApiClient
):
    mock_get_client.return_value = rest_client
    deployment = Deployment.update(
        deployment_config_params=DeploymentConfigParameters(
            cfg_dict={
                "name": "test-distributed",
                "services": {
                    "irisclassifier": {"scaling": {"max_replicas": 50}},
                    "preprocessing": {"instance_type": "t3-large"},
                },
            },
            service_name="irisclassifier",
        )
    )
    # assert expected schema
    assert deployment.cluster == "default_display_name"
    assert deployment.name == "test-distributed"
    config = deployment.get_config(refetch=False)
    assert config.access_type == AccessControl.PUBLIC
    assert config.envs == [EnvItemSchema(name="env_key", value="env_value")]
    assert config.services == {
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
    }
