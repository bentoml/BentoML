from __future__ import annotations

import json
import typing as t
import logging

import attr

from ..tag import Tag
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from .config import get_rest_api_client
from .config import default_context_name
from .config import default_kube_namespace
from .schemas import DeploymentMode
from .schemas import schema_to_json
from .schemas import DeploymentSchema
from .schemas import DeploymentListSchema
from .schemas import DeploymentTargetType
from .schemas import CreateDeploymentSchema
from .schemas import DeploymentTargetConfig
from .schemas import DeploymentTargetHPAConf
from .schemas import DeploymentTargetCanaryRule
from .schemas import DeploymentTargetRunnerConfig
from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)


@attr.define
class Resource:
    @classmethod
    def for_hpa_conf(cls, **kwargs) -> DeploymentTargetHPAConf:
        return bentoml_cattr.structure(kwargs, DeploymentTargetHPAConf)

    @classmethod
    def for_runner(cls, **kwargs) -> DeploymentTargetRunnerConfig:
        exclusive_api_server_key = {
            v for v in kwargs if v not in attr.fields_dict(DeploymentTargetRunnerConfig)
        }
        return bentoml_cattr.structure(
            {k: v for k, v in kwargs.items() if k not in exclusive_api_server_key},
            DeploymentTargetRunnerConfig,
        )

    @classmethod
    def for_api_server(cls, **kwargs) -> DeploymentTargetConfig:
        return bentoml_cattr.structure(kwargs, DeploymentTargetConfig)


class Deployment:
    @classmethod
    def _create_deployment(
        cls,
        context: str | None = None,
        cluster_name: str | None = None,
        *,
        create_deployment_schema: CreateDeploymentSchema,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        for target in create_deployment_schema.targets:
            res = yatai_rest_client.get_bento(target.bento_repository, target.bento)
            if res is None:
                raise BentoMLException(
                    f"Create deployment: {target.bento_repository}:{target.bento} does not exist"
                )
        res = yatai_rest_client.get_deployment(
            cluster_name,
            create_deployment_schema.kube_namespace,
            create_deployment_schema.name,
        )
        if res is not None:
            raise BentoMLException("Create deployment: Deployment already exists")
        res = yatai_rest_client.create_deployment(
            cluster_name, schema_to_json(create_deployment_schema)
        )
        if res is None:
            raise BentoMLException("Create deployment request failed")
        return res

    @classmethod
    def create(
        cls,
        deployment_name: str,
        bento_repository: Tag | str,
        description: str = None,
        cluster_name: str = default_context_name,
        kube_namespace: str = default_kube_namespace,
        resource_instance: str | None = None,
        cpu: str | None = None,
        gpu: str | None = None,
        memory: str | None = None,
        hpa_conf: DeploymentTargetHPAConf | None = None,
        runners_config: dict[str, DeploymentTargetRunnerConfig] | None = None,
        api_server_config: DeploymentTargetConfig | None = None,
        mode: DeploymentMode = DeploymentMode.Function,
        type: DeploymentTargetType = DeploymentTargetType.STABLE,
        context: str | None = None,
        do_not_deploy: bool = False,
        labels: t.List[dict[str, str]] | None = None,
        canary_rules: t.List[DeploymentTargetCanaryRule] | None = None,
    ) -> DeploymentSchema:
        if isinstance(bento_repository, str):
            bento_repository = bentoml_cattr.structure(bento_repository, Tag)
        dct = {
            "name": deployment_name,
            "kube_namespace": kube_namespace,
            "mode": mode,
            "labels": labels,
            "description": description,
            "do_not_deploy": do_not_deploy,
        }
        if api_server_config is None:
            dct["targets"] = [
                {
                    "type": type,
                    "bento_repository": bento_repository.name,
                    "bento": bento_repository.version,
                    "canary_rules": canary_rules,
                    "config": {
                        "resources": {},
                        "runners": runners_config,
                    },
                }
            ]

        else:
            api_server_config.runners = runners_config
            dct["targets"] = [
                {
                    "type": type,
                    "bento_repository": bento_repository.name,
                    "bento": bento_repository.version,
                    "canary_rules": canary_rules,
                    "config": bentoml_cattr.unstructure_attrs_asdict(api_server_config),
                }
            ]
        if cpu or gpu or memory:
            resource_spec = {"cpu": cpu, "gpu": gpu, "memory": memory}
            for target in dct["targets"]:
                if target["config"]["resources"] is None:
                    target["config"]["resources"] = {"requests": resource_spec}
                elif target["config"]["resources"].get("requests") is None:
                    target["config"]["resources"]["requests"] = resource_spec
                else:
                    for k, v in resource_spec.items():
                        if (
                            v is not None
                            and target["config"]["resources"]["requests"].get(k) is None
                        ):
                            target["config"]["resources"]["requests"][k] = v
                if target["config"].get("runners") is not None:
                    for _, config in target["config"]["runners"].items():
                        if config.get("resources") is None:
                            config["resources"] = {"requests": resource_spec}
                        elif config["resources"].get("requests") is None:
                            config["resources"]["requests"] = resource_spec
                        else:
                            for k, v in resource_spec.items():
                                if (
                                    v is not None
                                    and config["resources"]["requests"].get(k) is None
                                ):
                                    config["resources"]["requests"][k] = v

        if hpa_conf:
            hpa_conf_dct = bentoml_cattr.unstructure_attrs_asdict(hpa_conf)
            for target in dct["targets"]:
                if target["config"].get("hpa_conf") is None:
                    target["config"]["hpa_conf"] = hpa_conf_dct
                if target["config"]["runners"] is not None:
                    for _, config in target["config"]["runners"].items():
                        if config.get("hpa_conf") is None:
                            config["hpa_conf"] = hpa_conf_dct
        
        if resource_instance:
            for target in dct["targets"]:
                if target["config"].get("resource_instance") is None:
                    target["config"]["resource_instance"] = resource_instance
                if target["config"]["runners"] is not None:
                    for _, config in target["config"]["runners"].items():
                        if config.get("resource_instance") is None:
                            config["resource_instance"] = resource_instance


        create_deployment_schema = bentoml_cattr.structure(dct, CreateDeploymentSchema)
        logger.info(f"{create_deployment_schema.name} is created.")
        return cls._create_deployment(
            context=context,
            cluster_name=cluster_name,
            create_deployment_schema=create_deployment_schema,
        )

    @classmethod
    def list(
        cls, context: str | None = None, cluster_name: str | None = None
    ) -> DeploymentListSchema:

        yatai_rest_client = get_rest_api_client(context)
        if cluster_name:
            res = yatai_rest_client.get_deployment_list(cluster_name)
        else:
            res = yatai_rest_client.get_deployment_list(default_context_name)
        if res is None:
            raise BentoMLException("List deployments request failed")
        return res

    @classmethod
    def create_from_file(
        cls,
        path: str,
        context: str | None = None,
        cluster_name: str | None = None,
        path_context: str | None = None,
    ) -> DeploymentSchema:
        with open(path, "r"):
            data = json.load(resolve_user_filepath(path, path_context))

        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        deployment_schema = bentoml_cattr.structure(data, CreateDeploymentSchema)
        for target in deployment_schema.targets:
            res = yatai_rest_client.get_bento(target.bento_repository, target.bento)
            if res is None:
                raise BentoMLException(
                    f"Create deployment: {target.bento_repository}:{target.bento} does not exist"
                )
        res = yatai_rest_client.get_deployment(
            cluster_name, deployment_schema.kube_namespace, deployment_schema.name
        )
        if res is not None:
            raise BentoMLException("Create deployment: Deployment already exists")
        res = cls._create_deployment(cluster_name, deployment_schema)
        if res is None:
            raise BentoMLException("Create deployment request failed")
        return res

    @classmethod
    def get(
        cls,
        context: str | None = None,
        cluster_name: str | None = None,
        kubeNamespace: str | None = None,
        *,
        deploymentName: str,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        if kubeNamespace is None:
            kubeNamespace = default_kube_namespace
        res = yatai_rest_client.get_deployment(
            cluster_name, kubeNamespace, deploymentName
        )
        if res is None:
            raise BentoMLException("Get deployment request failed")
        return res
