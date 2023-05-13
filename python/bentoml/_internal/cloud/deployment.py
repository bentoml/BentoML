from __future__ import annotations

import json
import logging
import typing as t

import attr
from deepmerge.merger import Merger

from ...exceptions import BentoMLException
from ..tag import Tag
from ..utils import bentoml_cattr
from ..utils import first_not_none
from ..utils import resolve_user_filepath
from .config import get_rest_api_client
from .schemas import CreateDeploymentSchema
from .schemas import DeploymentListSchema
from .schemas import DeploymentMode
from .schemas import DeploymentSchema
from .schemas import DeploymentTargetCanaryRule
from .schemas import DeploymentTargetConfig
from .schemas import DeploymentTargetHPAConf
from .schemas import DeploymentTargetRunnerConfig
from .schemas import DeploymentTargetType
from .schemas import FullDeploymentSchema
from .schemas import UpdateDeploymentSchema

logger = logging.getLogger(__name__)

config_merger = Merger(
    # merge dicts
    type_strategies=[(dict, "merge")],
    # override all other types
    fallback_strategies=["override"],
    # override conflicting types
    type_conflict_strategies=["override"],
)


@attr.define
class Resource:
    @classmethod
    def for_hpa_conf(cls, **kwargs: t.Any) -> DeploymentTargetHPAConf:
        return bentoml_cattr.structure(kwargs, DeploymentTargetHPAConf)

    @classmethod
    def for_runner(cls, **kwargs: t.Any) -> DeploymentTargetRunnerConfig:
        exclusive_api_server_key = {
            v for v in kwargs if v not in attr.fields_dict(DeploymentTargetRunnerConfig)
        }
        return bentoml_cattr.structure(
            {k: v for k, v in kwargs.items() if k not in exclusive_api_server_key},
            DeploymentTargetRunnerConfig,
        )

    @classmethod
    def for_api_server(cls, **kwargs: t.Any) -> DeploymentTargetConfig:
        return bentoml_cattr.structure(kwargs, DeploymentTargetConfig)


class Deployment:
    @classmethod
    def _get_default_kube_namespace(
        cls,
        cluster_name: str,
        context: str | None = None,
    ) -> str:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.get_cluster(cluster_name)
        if not res:
            raise BentoMLException("Cannot get default kube namespace")
        return res.config.default_deployment_kube_namespace

    @classmethod
    def _get_default_cluster(cls, context: str | None = None) -> str:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.get_cluster_list(params={"count": 1})
        if not res:
            raise BentoMLException("Failed to get list of clusters.")
        if not res.items:
            raise BentoMLException("Cannot get default clusters.")
        return res.items[0].name

    @classmethod
    def _create_deployment(
        cls,
        create_deployment_schema: CreateDeploymentSchema,
        context: str | None = None,
        cluster_name: str | None = None,
    ) -> DeploymentSchema:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if create_deployment_schema.kube_namespace is None:
            create_deployment_schema.kube_namespace = cls._get_default_kube_namespace(
                cluster_name, context
            )
        for target in create_deployment_schema.targets:
            if (
                cloud_rest_client.get_bento(target.bento_repository, target.bento)
                is None
            ):
                raise BentoMLException(
                    f"Create deployment: {target.bento_repository}:{target.bento} does not exist"
                )
        if (
            cloud_rest_client.get_deployment(
                cluster_name,
                create_deployment_schema.kube_namespace,
                create_deployment_schema.name,
            )
            is not None
        ):
            raise BentoMLException("Create deployment: Deployment already exists")
        res = cloud_rest_client.create_deployment(
            cluster_name, create_deployment_schema
        )
        if res is None:
            raise BentoMLException("Create deployment request failed")
        logger.debug("Deployment Schema: %s", create_deployment_schema)
        return res

    @classmethod
    def _update_deployment(
        cls,
        deployment_name: str,
        update_deployment_schema: UpdateDeploymentSchema,
        kube_namespace: str | None = None,
        context: str | None = None,
        cluster_name: str | None = None,
    ) -> DeploymentSchema:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)
        for target in update_deployment_schema.targets:
            if (
                cloud_rest_client.get_bento(target.bento_repository, target.bento)
                is None
            ):
                raise BentoMLException(
                    f"Update deployment: {target.bento_repository}:{target.bento} does not exist"
                )
            cloud_rest_client.get_deployment(
                cluster_name,
                kube_namespace,
                deployment_name,
            )

        res = cloud_rest_client.update_deployment(
            cluster_name, kube_namespace, deployment_name, update_deployment_schema
        )
        if res is None:
            raise BentoMLException("Update deployment request failed")
        logger.debug("%s is created.", deployment_name)
        logger.debug("Deployment Schema: %s", update_deployment_schema)
        return res

    @classmethod
    def update(
        cls,
        deployment_name: str,
        bento: Tag | str | None = None,
        description: str | None = None,
        expose_endpoint: bool | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
        resource_instance: str | None = None,
        hpa_conf: DeploymentTargetHPAConf | None = None,
        runners_config: dict[str, DeploymentTargetRunnerConfig] | None = None,
        api_server_config: DeploymentTargetConfig | None = None,
        mode: DeploymentMode | None = None,
        type: DeploymentTargetType | None = None,
        context: str | None = None,
        labels: dict[str, str] | None = None,
        canary_rules: t.List[DeploymentTargetCanaryRule] | None = None,
        latest_bento: bool = False,
    ) -> DeploymentSchema:
        from bentoml import get as get_bento

        if mode is None:
            mode = DeploymentMode.Function
        if type is None:
            type = DeploymentTargetType.STABLE

        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)

        base_schema = cls.get(deployment_name, context, cluster_name, kube_namespace)
        # Deployment target always has length of 1
        if base_schema.latest_revision is None:
            raise BentoMLException(
                f"Deployment {deployment_name} has no latest revision"
            )
        deployment_target = base_schema.latest_revision.targets[0]

        if bento is None:
            # NOTE: bento.repository.name is the bento.name, and bento.name is the bento.version
            # from bentocloud to bentoml.Tag concept
            bento = deployment_target.bento.repository.name
        bento = Tag.from_taglike(bento)
        if latest_bento and bento.version is None or bento.version == "latest":
            bento = get_bento(bento).tag
        elif bento.version is None:
            bento.version = deployment_target.bento.name

        updated_config = bentoml_cattr.unstructure(deployment_target.config)
        if hpa_conf is not None:
            hpa_conf_dct = bentoml_cattr.unstructure(hpa_conf)
            if "hpa_conf" in updated_config:
                if updated_config["hpa_conf"] is None:
                    updated_config["hpa_conf"] = {}
                config_merger.merge(updated_config["hpa_conf"], hpa_conf_dct)
            if "runners" in updated_config and updated_config["runners"] is not None:
                for _, runner in updated_config["runners"].items():
                    if runner["hpa_conf"] is None:
                        runner["hpa_conf"] = {}
                    config_merger.merge(runner["hpa_conf"], hpa_conf_dct)
        if resource_instance is not None:
            updated_config["resource_instance"] = resource_instance
            if updated_config.get("runners") is not None:
                for runner in updated_config["runners"].values():
                    runner["resource_instance"] = resource_instance
        if expose_endpoint is not None:
            updated_config["enable_ingress"] = expose_endpoint

        if api_server_config is not None:
            if runners_config is not None:
                api_server_config.runners = runners_config
            config_merger.merge(
                updated_config, bentoml_cattr.unstructure(api_server_config)
            )
        elif runners_config is not None:
            config_merger.merge(
                updated_config,
                {
                    "runners": {
                        k: bentoml_cattr.unstructure(v)
                        for k, v in runners_config.items()
                    }
                },
            )

        dct_update: dict[str, t.Any] = {
            "mode": first_not_none(mode, base_schema.mode),
            "labels": first_not_none(
                [{"key": key, "value": value} for key, value in labels.items()]
                if labels
                else None,
                [bentoml_cattr.unstructure(i) for i in base_schema.labels],
            ),
            "description": description,
        }
        update_target = {
            "type": first_not_none(type, deployment_target.type),
            "bento": first_not_none(bento.version, deployment_target.bento.name),
            "bento_repository": first_not_none(
                bento.name, deployment_target.bento.repository.name
            ),
            "config": updated_config,
        }

        rules = first_not_none(
            [bentoml_cattr.unstructure(i) for i in canary_rules]
            if canary_rules
            else None,
            [bentoml_cattr.unstructure(i) for i in deployment_target.canary_rules]
            if deployment_target.canary_rules
            else None,
        )
        if rules:
            update_target["canary_rules"] = rules

        # update the target
        dct_update["targets"] = [update_target]

        return cls._update_deployment(
            deployment_name=deployment_name,
            update_deployment_schema=bentoml_cattr.structure(
                dct_update, UpdateDeploymentSchema
            ),
            context=context,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
        )

    @classmethod
    def create(
        cls,
        deployment_name: str,
        bento: Tag | str,
        description: str | None = None,
        expose_endpoint: bool | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
        resource_instance: str | None = None,
        hpa_conf: DeploymentTargetHPAConf | None = None,
        runners_config: dict[str, DeploymentTargetRunnerConfig] | None = None,
        api_server_config: DeploymentTargetConfig | None = None,
        mode: DeploymentMode | None = None,
        type: DeploymentTargetType | None = None,
        context: str | None = None,
        labels: dict[str, str] | None = None,
        canary_rules: t.List[DeploymentTargetCanaryRule] | None = None,
    ) -> DeploymentSchema:
        if mode is None:
            mode = DeploymentMode.Function
        if type is None:
            type = DeploymentTargetType.STABLE
        bento_tag = Tag.from_taglike(bento)

        dct: dict[str, t.Any] = {
            "name": deployment_name,
            "kube_namespace": kube_namespace,
            "mode": mode,
            "description": description,
        }
        if labels:
            dct["labels"] = [
                {"key": key, "value": value} for key, value in labels.items()
            ]
        if api_server_config is None:
            _config: dict[str, t.Any] = {
                "runners": {
                    k: bentoml_cattr.unstructure(v) for k, v in runners_config.items()
                }
                if runners_config
                else None,
            }
        else:
            api_server_config.runners = runners_config
            _config = bentoml_cattr.unstructure(api_server_config)

        create_target: dict[str, t.Any] = {
            "type": type,
            "bento_repository": bento_tag.name,
            "bento": bento_tag.version,
            "config": _config,
        }
        if canary_rules:
            create_target["canary_rules"] = [
                bentoml_cattr.unstructure(i) for i in canary_rules
            ]

        # Only change the value by the top-level param if it is not provided already in api_server_config or runner_config
        if hpa_conf:
            hpa_conf_dct = bentoml_cattr.unstructure(hpa_conf)
            _config_hpa_conf = _config.get("hpa_conf", None)
            if _config_hpa_conf is None:
                _config_hpa_conf = {}
            for k, v in hpa_conf_dct.items():
                if k not in _config_hpa_conf:
                    _config_hpa_conf[k] = v
                else:
                    logger.warning(
                        "Key %s is already set in API server config and will not be overwritten with hpa_conf.%s",
                        k,
                        k,
                    )
            _config["hpa_conf"] = _config_hpa_conf
            if "runners" in _config and _config["runners"] is not None:
                _runner_config = _config["runners"]
                for runner in _runner_config.values():
                    _runner_hpa_conf = runner.get("hpa_conf", None)
                    if _runner_hpa_conf is None:
                        _runner_hpa_conf = {}
                    for k, v in hpa_conf_dct.items():
                        if k not in _runner_hpa_conf:
                            _runner_hpa_conf[k] = v
                        else:
                            logger.warning(
                                "Key %s is already set in runner config and will not be overwritten with hpa_conf.%s",
                                k,
                                k,
                            )
                    runner["hpa_conf"] = _runner_hpa_conf

        if resource_instance:
            if "resource_instance" not in _config:
                _config["resource_instance"] = resource_instance
            if "runners" in _config and _config["runners"] is not None:
                for runner in _config["runners"].values():
                    if "resource_instance" not in runner:
                        runner["resource_instance"] = resource_instance

        if expose_endpoint is not None and _config.get("enable_ingress", None) is None:
            _config["enable_ingress"] = expose_endpoint

        # setup the create target itself
        dct["targets"] = [create_target]

        return cls._create_deployment(
            context=context,
            cluster_name=cluster_name,
            create_deployment_schema=bentoml_cattr.structure(
                dct, CreateDeploymentSchema
            ),
        )

    @classmethod
    def list(
        cls,
        context: str | None = None,
        cluster_name: str | None = None,
        query: str | None = None,
        search: str | None = None,
        count: int | None = None,
        start: int | None = None,
    ) -> DeploymentListSchema:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if query or start or count or search:
            params = {"start": start, "count": count, "search": search, "q": query}
            res = cloud_rest_client.get_deployment_list(cluster_name, **params)
            if res is None:
                raise BentoMLException("List deployments request failed")
            return res
        else:
            all_deployment = cloud_rest_client.get_deployment_list(cluster_name)
            if all_deployment is None:
                raise BentoMLException("List deployments request failed")
            return all_deployment

    @classmethod
    def create_from_file(
        cls,
        path_or_stream: str | t.TextIO,
        path_context: str | None = None,
        context: str | None = None,
    ) -> DeploymentSchema:
        if isinstance(path_or_stream, str):
            real_path = resolve_user_filepath(path_or_stream, path_context)
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
        else:
            # load the data from trusted stream
            data = json.load(path_or_stream)
        deployment_schema = bentoml_cattr.structure(data, FullDeploymentSchema)
        return cls._create_deployment(
            create_deployment_schema=deployment_schema,
            context=context,
            cluster_name=deployment_schema.cluster_name,
        )

    @classmethod
    def update_from_file(
        cls,
        path_or_stream: str | t.TextIO,
        path_context: str | None = None,
        context: str | None = None,
    ) -> DeploymentSchema:
        if isinstance(path_or_stream, str):
            real_path = resolve_user_filepath(path_or_stream, path_context)
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
        else:
            data = json.load(path_or_stream)
        deployment_schema = bentoml_cattr.structure(data, FullDeploymentSchema)
        return cls._update_deployment(
            deployment_name=deployment_schema.name,
            update_deployment_schema=deployment_schema,
            context=context,
            cluster_name=deployment_schema.cluster_name,
            kube_namespace=deployment_schema.kube_namespace,
        )

    @classmethod
    def get(
        cls,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> DeploymentSchema:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)
        res = cloud_rest_client.get_deployment(
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
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)
        res = cloud_rest_client.get_deployment(
            cluster_name,
            kube_namespace,
            deployment_name,
        )
        if res is None:
            raise BentoMLException("Delete deployment: Deployment does not exist")

        res = cloud_rest_client.delete_deployment(
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
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)
        res = cloud_rest_client.get_deployment(
            cluster_name,
            kube_namespace,
            deployment_name,
        )
        if res is None:
            raise BentoMLException("Teminate deployment: Deployment does not exist")
        res = cloud_rest_client.terminate_deployment(
            cluster_name, kube_namespace, deployment_name
        )
        if res is None:
            raise BentoMLException("Terminate deployment request failed")
        return res
