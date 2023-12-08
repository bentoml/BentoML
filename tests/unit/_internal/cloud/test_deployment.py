from __future__ import annotations

import typing as t
from datetime import datetime
from unittest.mock import patch

import attr
import pytest

from bentoml._internal.cloud.schemas import BentoFullSchema
from bentoml._internal.cloud.schemas import BentoImageBuildStatus
from bentoml._internal.cloud.schemas import BentoManifestSchema
from bentoml._internal.cloud.schemas import BentoRepositorySchema
from bentoml._internal.cloud.schemas import BentoUploadStatus
from bentoml._internal.cloud.schemas import ClusterSchema
from bentoml._internal.cloud.schemas import CreateDeploymentSchema
from bentoml._internal.cloud.schemas import CreateDeploymentTargetSchema
from bentoml._internal.cloud.schemas import DeploymentMode
from bentoml._internal.cloud.schemas import DeploymentRevisionSchema
from bentoml._internal.cloud.schemas import DeploymentRevisionStatus
from bentoml._internal.cloud.schemas import DeploymentSchema
from bentoml._internal.cloud.schemas import DeploymentStatus
from bentoml._internal.cloud.schemas import DeploymentTargetCanaryRule
from bentoml._internal.cloud.schemas import DeploymentTargetCanaryRuleType
from bentoml._internal.cloud.schemas import DeploymentTargetConfig
from bentoml._internal.cloud.schemas import DeploymentTargetHPAConf
from bentoml._internal.cloud.schemas import DeploymentTargetRunnerConfig
from bentoml._internal.cloud.schemas import DeploymentTargetSchema
from bentoml._internal.cloud.schemas import DeploymentTargetType
from bentoml._internal.cloud.schemas import LabelItemSchema
from bentoml._internal.cloud.schemas import ResourceType
from bentoml._internal.cloud.schemas import UpdateDeploymentSchema
from bentoml._internal.cloud.schemas import UserSchema
from bentoml.cloud import BentoCloudClient
from bentoml.cloud import Resource

if t.TYPE_CHECKING:
    from unittest.mock import MagicMock


def f_create(
    create_deployment_schema: CreateDeploymentSchema,
    context: str | None = None,
    cluster_name: str | None = None,
):
    return create_deployment_schema


def f_update(
    deployment_name: str,
    update_deployment_schema: UpdateDeploymentSchema,
    kube_namespace: str | None = None,
    context: str | None = None,
    cluster_name: str | None = None,
):
    return update_deployment_schema


@pytest.fixture(name="get_schema", scope="function")
def fixture_get_schema() -> DeploymentSchema:
    user = UserSchema(name="", email="", first_name="", last_name="")
    return DeploymentSchema(
        latest_revision=DeploymentRevisionSchema(
            targets=[
                DeploymentTargetSchema(
                    type=DeploymentTargetType.STABLE,
                    bento=BentoFullSchema(
                        uid="",
                        created_at=datetime(2023, 5, 25),
                        updated_at=None,
                        deleted_at=None,
                        name="12345",
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
                            name="",
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
                            name="iris_classifier",
                            resource_type=ResourceType.BENTO_REPOSITORY,
                            labels=[],
                            description="",
                            latest_bento="",
                        ),
                    ),
                    config=DeploymentTargetConfig(
                        resource_instance="t3-micro",
                        enable_ingress=True,
                        hpa_conf=DeploymentTargetHPAConf(
                            min_replicas=2, max_replicas=10
                        ),
                        runners={
                            "runner1": DeploymentTargetRunnerConfig(
                                resource_instance="t3-small",
                                hpa_conf=DeploymentTargetHPAConf(
                                    min_replicas=3, max_replicas=10
                                ),
                            ),
                            "runner2": DeploymentTargetRunnerConfig(
                                resource_instance="t3-medium",
                                hpa_conf=DeploymentTargetHPAConf(
                                    min_replicas=5, max_replicas=10
                                ),
                            ),
                        },
                    ),
                    canary_rules=[],
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


@pytest.fixture(scope="function", name="cloudclient")
def fixture_cloudclient() -> BentoCloudClient:
    return BentoCloudClient()


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create

    res = cloudclient.deployment.create(
        deployment_name="test-xxx", bento="iris_classifier:dqjxjyx2vweogcvj"
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_canary_rules(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    rules = [
        DeploymentTargetCanaryRule(DeploymentTargetCanaryRuleType.WEIGHT, 3, "", "", "")
    ]
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        canary_rules=rules,
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(),
                canary_rules=rules,
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_labels(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create

    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        labels={"user": "steve"},
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
        labels=[LabelItemSchema("user", "steve")],
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_resource_instance(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create

    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        resource_instance="test-instance",
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(resource_instance="test-instance"),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_resource_instance_runner(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    runner = Resource.for_runner(enable_debug_mode=True)

    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        resource_instance="test-instance",
        runners_config={"runner": runner},
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(
                    resource_instance="test-instance",
                    runners={
                        "runner": DeploymentTargetRunnerConfig(
                            resource_instance="test-instance", enable_debug_mode=True
                        )
                    },
                ),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_resource_instance_api_server(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    api_server = Resource.for_api_server(enable_ingress=True)

    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        resource_instance="test-resource",
        api_server_config=api_server,
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(
                    resource_instance="test-resource", enable_ingress=True
                ),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_api_server(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    api_server_conf = Resource.for_api_server(resource_instance="t3-micro")
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        api_server_config=api_server_conf,
    )

    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(resource_instance="t3-micro"),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_hpa_conf(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    hpa_conf = Resource.for_hpa_conf(min_replicas=2, max_replicas=10)
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        hpa_conf=hpa_conf,
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(hpa_conf=hpa_conf),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_runner(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    runner = Resource.for_runner(resource_instance="t3-micro", enable_debug_mode=True)
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        runners_config={"runner1": runner},
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(runners={"runner1": runner}),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_runner_hpa_conf(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    hpa_conf = Resource.for_hpa_conf(min_replicas=2, max_replicas=10)
    runner = Resource.for_runner(resource_instance="t3-micro", enable_debug_mode=True)
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        runners_config={"runner1": runner},
        hpa_conf=hpa_conf,
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(
                    hpa_conf=hpa_conf,
                    runners={
                        "runner1": DeploymentTargetRunnerConfig(
                            resource_instance="t3-micro",
                            hpa_conf=hpa_conf,
                            enable_debug_mode=True,
                        )
                    },
                ),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_api_server_runner(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    api_server = Resource.for_api_server(
        resource_instance="t3-micro", enable_stealing_traffic_debug_mode=True
    )
    runner = Resource.for_runner(resource_instance="t3-micro", enable_debug_mode=True)
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        runners_config={"runner1": runner},
        api_server_config=api_server,
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(
                    resource_instance="t3-micro",
                    enable_stealing_traffic_debug_mode=True,
                    runners={
                        "runner1": DeploymentTargetRunnerConfig(
                            resource_instance="t3-micro", enable_debug_mode=True
                        )
                    },
                ),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_api_server_hpa_conf(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    api_server = Resource.for_api_server(resource_instance="t3-micro")
    hpa_conf = Resource.for_hpa_conf(min_replicas=2, max_replicas=10)
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        api_server_config=api_server,
        hpa_conf=hpa_conf,
    )
    assert res == CreateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(
                    resource_instance="t3-micro", hpa_conf=hpa_conf
                ),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@patch("bentoml._internal.cloud.deployment.Deployment._create_deployment")
def test_create_deployment_api_server_runner_hpa_conf(
    mock_create_deployment: MagicMock, cloudclient: BentoCloudClient
):
    mock_create_deployment.side_effect = f_create
    api_server = Resource.for_api_server(resource_instance="t3-micro")
    runner = Resource.for_runner(
        resource_instance="t3-small", hpa_conf={"min_replicas": 3}
    )
    runner2 = Resource.for_runner(
        resource_instance="t3-medium", hpa_conf={"min_replicas": 5}
    )
    hpa_conf = Resource.for_hpa_conf(min_replicas=2, max_replicas=10)
    res = cloudclient.deployment.create(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        api_server_config=api_server,
        hpa_conf=hpa_conf,
        runners_config={"runner1": runner, "runner2": runner2},
        expose_endpoint=True,
        labels={"user": "steve"},
    )
    assert res == CreateDeploymentSchema(
        labels=[LabelItemSchema("user", "steve")],
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="dqjxjyx2vweogcvj",
                config=DeploymentTargetConfig(
                    resource_instance="t3-micro",
                    enable_ingress=True,
                    hpa_conf=hpa_conf,
                    runners={
                        "runner1": DeploymentTargetRunnerConfig(
                            resource_instance="t3-small",
                            hpa_conf=DeploymentTargetHPAConf(
                                min_replicas=3, max_replicas=10
                            ),
                        ),
                        "runner2": DeploymentTargetRunnerConfig(
                            resource_instance="t3-medium",
                            hpa_conf=DeploymentTargetHPAConf(
                                min_replicas=5, max_replicas=10
                            ),
                        ),
                    },
                ),
            )
        ],
        mode=DeploymentMode.Function,
        name="test-xxx",
    )


@pytest.fixture(name="update_schema", scope="function")
def fixture_update_schema() -> UpdateDeploymentSchema:
    return UpdateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="12345",
                config=DeploymentTargetConfig(
                    resource_instance="t3-micro",
                    enable_ingress=True,
                    hpa_conf=DeploymentTargetHPAConf(min_replicas=2, max_replicas=10),
                    runners={
                        "runner1": DeploymentTargetRunnerConfig(
                            resource_instance="t3-small",
                            hpa_conf=DeploymentTargetHPAConf(
                                min_replicas=3, max_replicas=10
                            ),
                        ),
                        "runner2": DeploymentTargetRunnerConfig(
                            resource_instance="t3-medium",
                            hpa_conf=DeploymentTargetHPAConf(
                                min_replicas=5, max_replicas=10
                            ),
                        ),
                    },
                ),
            )
        ],
        mode=DeploymentMode.Function,
        labels=[],
    )


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_bento(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        bento="iris_classifier:dqjxjyx2vweogcvj",
        cluster_name="",
        kube_namespace="",
    )
    update_schema.targets[0].bento = "dqjxjyx2vweogcvj"
    assert res == update_schema


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_runner(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    new_runnner = Resource.for_runner(
        resource_instance="new-resource", hpa_conf={"min_replicas": 6}
    )
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        cluster_name="",
        kube_namespace="",
        runners_config={"runner1": new_runnner},
    )
    update_schema.targets[0].config.runners["runner1"].hpa_conf.min_replicas = 6
    update_schema.targets[0].config.runners[
        "runner1"
    ].resource_instance = "new-resource"
    assert res == update_schema


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_runner_hpa_conf(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    hpa_conf = Resource.for_hpa_conf(min_replicas=5)
    new_runnner = Resource.for_runner(
        resource_instance="new-resource", hpa_conf={"min_replicas": 7}
    )
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        cluster_name="",
        kube_namespace="",
        runners_config={"runner1": new_runnner},
        hpa_conf=hpa_conf,
    )
    update_schema.targets[0].config.hpa_conf.min_replicas = 5
    for k, v in update_schema.targets[0].config.runners.items():
        if k == "runner1":
            v.hpa_conf.min_replicas = 7
            v.resource_instance = "new-resource"
        else:
            v.hpa_conf.min_replicas = 5
    assert res == update_schema


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_api_server(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    api_server = Resource.for_api_server(
        enable_ingress=False, hpa_conf={"min_replicas": 5}
    )
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        cluster_name="",
        kube_namespace="",
        api_server_config=api_server,
    )
    update_schema.targets[0].config.hpa_conf.min_replicas = 5
    update_schema.targets[0].config.enable_ingress = False
    assert res == update_schema


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_api_server_hpa_conf(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    api_server = Resource.for_api_server(hpa_conf={"min_replicas": 9})
    hpa_conf = Resource.for_hpa_conf(min_replicas=8)
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        cluster_name="",
        kube_namespace="",
        api_server_config=api_server,
        hpa_conf=hpa_conf,
    )
    UpdateDeploymentSchema(
        targets=[
            CreateDeploymentTargetSchema(
                type=DeploymentTargetType.STABLE,
                bento_repository="iris_classifier",
                bento="12345",
                config=DeploymentTargetConfig(
                    resource_instance="t3-micro",
                    enable_ingress=True,
                    hpa_conf=DeploymentTargetHPAConf(min_replicas=2, max_replicas=10),
                    runners={
                        "runner1": DeploymentTargetRunnerConfig(
                            resource_instance="t3-small",
                            hpa_conf=DeploymentTargetHPAConf(
                                min_replicas=3, max_replicas=10
                            ),
                        ),
                        "runner2": DeploymentTargetRunnerConfig(
                            resource_instance="t3-medium",
                            hpa_conf=DeploymentTargetHPAConf(
                                min_replicas=5, max_replicas=10
                            ),
                        ),
                    },
                ),
            )
        ],
        mode=DeploymentMode.Function,
        labels=[],
    )
    update_schema.targets[0].config.hpa_conf.min_replicas = 9
    for _, v in update_schema.targets[0].config.runners.items():
        v.hpa_conf.min_replicas = 8
    assert res == update_schema


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_resource_instance(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        cluster_name="",
        kube_namespace="",
        resource_instance="test-resource",
    )
    update_schema.targets[0].config.resource_instance = "test-resource"
    for _, v in update_schema.targets[0].config.runners.items():
        v.resource_instance = "test-resource"
    assert res == update_schema


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_labels(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        cluster_name="",
        kube_namespace="",
        labels={"user": "steve"},
    )
    assert res == attr.evolve(update_schema, labels=[LabelItemSchema("user", "steve")])


@patch("bentoml._internal.cloud.deployment.Deployment.get")
@patch("bentoml._internal.cloud.deployment.Deployment._update_deployment")
def test_update_deployment_canary_rules(
    mock_update_deployment: MagicMock,
    mock_get: MagicMock,
    update_schema: UpdateDeploymentSchema,
    get_schema: DeploymentSchema,
    cloudclient: BentoCloudClient,
):
    mock_update_deployment.side_effect = f_update
    mock_get.return_value = get_schema
    rules = [
        DeploymentTargetCanaryRule(DeploymentTargetCanaryRuleType.WEIGHT, 3, "", "", "")
    ]
    res = cloudclient.deployment.update(
        deployment_name="test-xxx",
        cluster_name="",
        kube_namespace="",
        canary_rules=rules,
    )
    update_schema.targets[0].canary_rules = rules
    assert res == update_schema
