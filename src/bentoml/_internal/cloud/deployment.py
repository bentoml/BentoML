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
from .schemas import UpdateDeploymentSchema
from .schemas import DeploymentTargetHPAConf
from .schemas import DeploymentTargetCanaryRule
from .schemas import DeploymentTargetRunnerConfig
from deepmerge.merger import Merger
from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

config_merger = Merger(
    # merge dicts
    type_strategies=[(dict, "merge")],
    # override all other types
    fallback_strategies=["override"],
    # override conflicting types
    type_conflict_strategies=["override"],
)

def delete_none(dct):
    """Delete None values recursively from all of the dictionaries"""
    for key, value in list(dct.items()):
        if isinstance(value, dict):
            delete_none(value)
        elif value is None:
            del dct[key]
        elif isinstance(value, list):
            for v_i in value:
                if isinstance(v_i, dict):
                    delete_none(v_i)

    return dct

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
        if kwargs.get("resources") is None:
            kwargs["resources"] = {}
        return bentoml_cattr.structure(kwargs, DeploymentTargetConfig)


class Deployment:
    @classmethod
    def _swap_schema_value(cls, key: str, value: t.Any, dct: dict):
        if dct.get(key) is None:
            dct[key] = value
        if dct["runners"] is not None:
            for _, config in dct["runners"].items():
                if config.get(key) is None:
                    config[key] = value

    @classmethod
    def _create_deployment(
        cls,
        create_deployment_schema: CreateDeploymentSchema,
        context: str | None = None,
        cluster_name: str | None = None,
    ) -> DeploymentSchema:
        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        for target in create_deployment_schema.targets:
            if (
                yatai_rest_client.get_bento(target.bento_repository, target.bento)
                is None
            ):
                raise BentoMLException(
                    f"Create deployment: {target.bento_repository}:{target.bento} does not exist"
                )
        if (
            yatai_rest_client.get_deployment(
                cluster_name,
                create_deployment_schema.kube_namespace,
                create_deployment_schema.name,
            )
            is not None
        ):
            raise BentoMLException("Create deployment: Deployment already exists")
        res = yatai_rest_client.create_deployment(
            cluster_name, schema_to_json(create_deployment_schema)
        )
        if res is None:
            raise BentoMLException("Create deployment request failed")
        return res

    @classmethod
    def _update_deployment(
        cls,
        deployment_name: str,
        kube_namespace: str,
        update_deployment_schema: UpdateDeploymentSchema,
        context: str | None = None,
        cluster_name: str | None = None,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        for target in update_deployment_schema.targets:
            if (
                yatai_rest_client.get_bento(target.bento_repository, target.bento)
                is None
            ):
                raise BentoMLException(
                    f"Create deployment: {target.bento_repository}:{target.bento} does not exist"
                )
            yatai_rest_client.get_deployment(
                cluster_name,
                kube_namespace,
                deployment_name,
            )

        res = yatai_rest_client.update_deployment(
            cluster_name, kube_namespace, deployment_name, update_deployment_schema
        )
        if res is None:
            raise BentoMLException("Update deployment request failed")
        return res

    @classmethod
    def update(
        cls,
        deployment_name: str,
        bento_repository: Tag | str | None = None,
        description: str = None,
        cluster_name: str = default_context_name,
        resource_instance: str | None = None,
        kube_namespace: str = default_kube_namespace,
        hpa_conf: DeploymentTargetHPAConf | None = None,
        runners_config: dict[str, DeploymentTargetRunnerConfig] | None = None,
        api_server_config: DeploymentTargetConfig | None = None,
        mode: DeploymentMode | None = None,
        type: DeploymentTargetType | None = None,
        context: str | None = None,
        labels: t.List[dict[str, str]] | None = None,
        canary_rules: t.List[DeploymentTargetCanaryRule] | None = None,
    ):

        if bento_repository:
            bento_repository = bento_repository.from_taglike(bento_repository)

        original_deployment_schema = cls.get(
            deployment_name, context, cluster_name, kube_namespace
        )
        # Deployment target always has length of 1
        deployment_target = original_deployment_schema.latest_revision.targets[0]
        dct_orig = {
            "mode": mode if mode else original_deployment_schema.mode,
            "labels": labels if labels else original_deployment_schema.labels,
            "description": description,
            "targets": [
                {
                    "type": type if type else deployment_target.type,
                    "bento": bento_repository.version
                    if bento_repository
                    else deployment_target.bento.name,
                    "bento_repository": bento_repository.name
                    if bento_repository
                    else deployment_target.bento.repository.name,
                    "canary_rules": canary_rules
                    if canary_rules
                    else deployment_target.canary_rules,
                    "config": bentoml_cattr.unstructure_attrs_asdict(deployment_target.config)
                }
            ],
        }

        dct_config = {}
        if api_server_config is None:
            dct_config["runners"] = {k:bentoml_cattr.unstructure_attrs_asdict(v) for k,v in runners_config.items()} if runners_config else None

        else:
            api_server_config.runners = runners_config
            dct_config = bentoml_cattr.unstructure_attrs_asdict(api_server_config)

        if hpa_conf:
            hpa_conf_dct = bentoml_cattr.unstructure_attrs_asdict(hpa_conf)
            cls._swap_schema_value("hpa_conf",hpa_conf_dct,dct_config)
            cls._swap_schema_value("hpa_conf",hpa_conf_dct, dct_orig["targets"][0]["config"])

        if resource_instance:
            cls._swap_schema_value("resource_instance",resource_instance,dct_config)
            cls._swap_schema_value("resource_instance",resource_instance, dct_orig["targets"][0]["config"])
        delete_none(dct_config)
        if dct_config.get('resources') == {}:
            del dct_config['resources']
        config_merger.merge(dct_orig["targets"][0]["config"],dct_config)
        cls._update_deployment(
            deployment_name,
            kube_namespace,
            bentoml_cattr.structure(dct_orig, UpdateDeploymentSchema),
            context,
            cluster_name,
        )

    @classmethod
    def create(
        cls,
        deployment_name: str,
        bento_repository: Tag | str,
        description: str = None,
        cluster_name: str = default_context_name,
        kube_namespace: str = default_kube_namespace,
        resource_instance: str | None = None,
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
        bento_repository = Tag.from_taglike(bento_repository)
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
                        "runners": {k:bentoml_cattr.unstructure_attrs_asdict(v) for k,v in runners_config.items()} if runners_config else None,
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

        if hpa_conf:
            hpa_conf_dct = bentoml_cattr.unstructure_attrs_asdict(hpa_conf)
            cls._swap_schema_value("hpa_conf",hpa_conf_dct,dct["targets"][0]["config"])

        if resource_instance:
            cls._swap_schema_value("resource_instance",resource_instance,dct["targets"][0]["config"])

        create_deployment_schema = bentoml_cattr.structure(dct, CreateDeploymentSchema)
        logger.debug("%s is created.", create_deployment_schema.name)
        logger.debug("Deployment Schema: %s", create_deployment_schema)
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
        if cluster_name is None:
            cluster_name = default_context_name
        res = yatai_rest_client.get_deployment_list(cluster_name)
        if res is None:
            raise BentoMLException("List deployments request failed")
        return res

    @classmethod
    def create_from_file(
        cls,
        path: str,
        path_context: str | None = None,
        context: str | None = None,
        cluster_name: str | None = None,
    ) -> DeploymentSchema:
        real_path = resolve_user_filepath(path, path_context)
        try:
            with open(real_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise ValueError(f"File not found: {real_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file: {real_path}\n{e}")
        except Exception as e:
            raise ValueError(
                f"An error occurred while reading the file: {real_path}\n{e}"
            )
        if cluster_name is None:
            cluster_name = default_context_name
        deployment_schema = bentoml_cattr.structure(data, CreateDeploymentSchema)
        return cls._create_deployment(
            create_deployment_schema=deployment_schema,
            context=context,
            cluster_name=cluster_name,
        )

    @classmethod
    def get(
        cls,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> DeploymentSchema:
        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        if kube_namespace is None:
            kube_namespace = default_kube_namespace
        res = yatai_rest_client.get_deployment(
            cluster_name, kube_namespace, deployment_name
        )
        if res is None:
            raise BentoMLException("Get deployment request failed")
        return res

    @classmethod
    def delete(
        cls,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        if kube_namespace is None:
            kube_namespace = default_kube_namespace
        res = yatai_rest_client.get_deployment(
            cluster_name,
            kube_namespace,
            deployment_name,
        )
        if res is None:
            raise BentoMLException("Delete deployment: Deployment does not exist")

        res = yatai_rest_client.delete_deployment(
            cluster_name, kube_namespace, deployment_name
        )
        if res is None:
            raise BentoMLException("Delete deployment request failed")
        return res

    @classmethod
    def terminate(
        cls,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = default_context_name
        if kube_namespace is None:
            kube_namespace = default_kube_namespace
        res = yatai_rest_client.get_deployment(
            cluster_name,
            kube_namespace,
            deployment_name,
        )
        if res is None:
            raise BentoMLException("Teminate deployment: Deployment does not exist")
        res = yatai_rest_client.terminate_deployment(
            cluster_name, kube_namespace, deployment_name
        )
        if res is None:
            raise BentoMLException("Terminate deployment request failed")
        return res
