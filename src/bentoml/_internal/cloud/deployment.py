from __future__ import annotations

import typing as t

import attr

from ..tag import Tag
from ..utils import bentoml_cattr
from .config import get_rest_api_client
from .config import default_context_name
from .config import default_kube_namespace
from .schemas import schema_to_json
from .schemas import LabelItemSchema
from .schemas import DeploymentSchema
from .schemas import schema_from_json
from .schemas import DeploymentStrategy
from .schemas import DeploymentListSchema
from .schemas import DeploymentTargetType
from .schemas import TrafficControlConfig
from .schemas import CreateDeploymentSchema
from .schemas import DeploymentTargetConfig
from .schemas import DeploymentTargetHPAConf
from .schemas import DeploymentTargetResources
from .schemas import CreateDeploymentTargetSchema
from .schemas import DeploymentTargetRunnerConfig
from .schemas import RunnerBentoDeploymentOverrides
from ...exceptions import BentoMLException


@attr.define
class DeploymentConfig:
    hpa_conf: DeploymentTargetHPAConf = attr.field(
        converter=(
            lambda x: bentoml_cattr.structure(x, DeploymentTargetHPAConf)
            if type(x) is dict
            else x
        )
    )
    resources: t.Optional[DeploymentTargetResources] = attr.field(
        converter=(
            lambda x: bentoml_cattr.structure(x, DeploymentTargetResources)
            if type(x) is dict
            else x
        )
    )
    resource_instance: t.Optional[str] = attr.field(default=None)
    envs: t.Optional[t.List[LabelItemSchema]] = attr.field(default=None)
    enable_stealing_traffic_debug_mode: t.Optional[bool] = attr.field(
        default=None, converter=attr.converters.default_if_none(False)
    )
    enable_debug_mode: t.Optional[bool] = attr.field(
        default=None, converter=attr.converters.default_if_none(False)
    )
    enable_debug_pod_receive_production_traffic: t.Optional[bool] = attr.field(
        default=None, converter=attr.converters.default_if_none(False)
    )
    deployment_strategy: t.Optional[DeploymentStrategy] = attr.field(
        default=None,
        converter=attr.converters.default_if_none(DeploymentStrategy.RollingUpdate),
    )
    bento_deployment_overrides: t.Optional[RunnerBentoDeploymentOverrides] = attr.field(
        default=None,
        converter=(
            lambda x: bentoml_cattr.structure(x, DeploymentTargetHPAConf)
            if type(x) is dict
            else x
        ),
    )
    traffic_control: t.Optional[TrafficControlConfig] = attr.field(
        default=TrafficControlConfig(timeout=60),
        converter=(
            lambda x: bentoml_cattr.structure(x, TrafficControlConfig)
            if type(x) is t.Dict
            else x
        ),
    )
    deployment_cold_start_wait_timeout: t.Optional[int] = attr.field(default=None)


class deployment:
    def list_deployment(
        self, context: str | None = None, clusterName: str | None = None
    ) -> DeploymentListSchema:

        yatai_rest_client = get_rest_api_client(context)
        if clusterName:
            res = yatai_rest_client.get_deployment_list(clusterName)
        else:
            res = yatai_rest_client.get_deployment_list(default_context_name)
        if res is None:
            raise BentoMLException("List deployments request failed")
        return res

    def create_deployment_by_json(
        self,
        context: str | None = None,
        clusterName: str | None = None,
        *,
        json_content: str,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if clusterName is None:
            clusterName = default_context_name
        deployment_schema = schema_from_json(json_content, CreateDeploymentSchema)
        for target in deployment_schema.targets:
            res = yatai_rest_client.get_bento(target.bento_repository, target.bento)
            if res is None:
                raise BentoMLException(
                    f"Create deployment: {target.bento_repository}:{target.bento} does not exist"
                )
        res = yatai_rest_client.get_deployment(
            clusterName, deployment_schema.kube_namespace, deployment_schema.name
        )
        if res is not None:
            raise BentoMLException("Create deployment: Deployment already exists")
        res = yatai_rest_client.create_deployment(clusterName, json_content)
        if res is None:
            raise BentoMLException("Create deployment request failed")
        return res

    def create_deployment(
        self,
        deployment_name: str,
        bento_repository: Tag,
        runner_config: dict[str, DeploymentConfig],
        enable_ingress: bool,
        mode="function",
        type=DeploymentTargetType.STABLE,
        context: str | None = None,
        clusterName: str | None = None,
        kube_namespace: str | None = None,
        api_server_config: DeploymentConfig | None = None,
        bento_request_overrides: dict[str, t.Any] | None = None,
        do_not_deploy: bool = False,
        description: str = "",
        labels: t.Dict[str, str] | None = None,
    ):
        dct = {}
        for key, one_runner_config in runner_config.items():
            dct[key] = DeploymentTargetRunnerConfig(
                hpa_conf=one_runner_config.hpa_conf,
                resources=one_runner_config.resources,
                resource_instance=one_runner_config.resource_instance,
                envs=one_runner_config.envs,
                enable_stealing_traffic_debug_mode=one_runner_config.enable_stealing_traffic_debug_mode,
                enable_debug_mode=one_runner_config.enable_debug_mode,
                enable_debug_pod_receive_production_traffic=one_runner_config.enable_debug_pod_receive_production_traffic,
                deployment_strategy=one_runner_config.deployment_strategy,
                bento_deployment_overrides=one_runner_config.bento_deployment_overrides,
                traffic_control=one_runner_config.traffic_control,
                deployment_cold_start_wait_timeout=one_runner_config.deployment_cold_start_wait_timeout,
            )

        deploy_target = DeploymentTargetConfig(
            hpa_conf=api_server_config.hpa_conf,
            resources=api_server_config.resources,
            resource_instance=api_server_config.resource_instance,
            envs=api_server_config.envs,
            enable_stealing_traffic_debug_mode=api_server_config.enable_stealing_traffic_debug_mode,
            enable_debug_mode=api_server_config.enable_debug_mode,
            enable_debug_pod_receive_production_traffic=api_server_config.enable_debug_pod_receive_production_traffic,
            deployment_strategy=api_server_config.deployment_strategy,
            bento_deployment_overrides=api_server_config.bento_deployment_overrides,
            traffic_control=api_server_config.traffic_control,
            deployment_cold_start_wait_timeout=api_server_config.deployment_cold_start_wait_timeout,
            enable_ingress=enable_ingress,
            runners=dct,
            bento_request_overrides=bento_request_overrides,
        )

        create_deploy_target = CreateDeploymentTargetSchema(
            type=type,
            bento_repository=bento_repository.name,
            bento=bento_repository.version,
            config=deploy_target,
        )
        create_deploy_schema = CreateDeploymentSchema(
            mode=mode,
            name=deployment_name,
            kube_namespace=kube_namespace,
            targets=[create_deploy_target],
            labels=[bentoml_cattr.structure(i) for i in labels]
            if labels is not None
            else None,
            do_not_deploy=do_not_deploy,
            description=description,
        )
        print(schema_to_json(deploy_target))
        print(schema_to_json(create_deploy_target))
        print(schema_to_json(create_deploy_schema))
        self.create_deployment_by_json(
            context, clusterName, json_content=schema_to_json(create_deploy_schema)
        )

    def get_deployment(
        self,
        context: str | None = None,
        clusterName: str | None = None,
        kubeNamespace: str | None = None,
        *,
        deploymentName: str,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if clusterName is None:
            clusterName = default_context_name
        if kubeNamespace is None:
            kubeNamespace = default_kube_namespace
        res = yatai_rest_client.get_deployment(
            clusterName, kubeNamespace, deploymentName
        )
        if res is None:
            raise BentoMLException("Get deployment request failed")
        return res
