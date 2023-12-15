from __future__ import annotations

import logging
import typing as t

import attr
import yaml
from deepmerge.merger import Merger
from simple_di import Provide
from simple_di import inject

if t.TYPE_CHECKING:
    from bentoml._internal.bento.bento import BentoStore
    from bentoml._internal.cloud.bentocloud import BentoCloudClient

from ...exceptions import BentoMLException
from ..configuration.containers import BentoMLContainer
from ..tag import Tag
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from .config import get_rest_api_client
from .schemas.modelschemas import ApiServerBentoFunctionOverrides
from .schemas.modelschemas import DeploymentTargetConfigV2
from .schemas.modelschemas import DeploymentTargetHPAConf
from .schemas.schemasv1 import DeploymentListSchema
from .schemas.schemasv1 import DeploymentSchema
from .schemas.schemasv2 import CreateDeploymentSchema as CreateDeploymentSchemaV2
from .schemas.schemasv2 import ExtraDeploymentOverrides
from .schemas.schemasv2 import UpdateDeploymentSchema as UpdateDeploymentSchemaV2

logger = logging.getLogger(__name__)

config_merger = Merger(
    # merge dicts
    type_strategies=[(dict, "merge")],
    # override all other types
    fallback_strategies=["override"],
    # override conflicting types
    type_conflict_strategies=["override"],
)


def deployment_target_schema_to_v2_update_schema(
    base_schema: DeploymentSchema,
) -> UpdateDeploymentSchemaV2:
    if base_schema.latest_revision is None:
        raise BentoMLException(f"Deployment {base_schema.name} has no latest revision")
    target_schema = base_schema.latest_revision.targets[0]
    res = UpdateDeploymentSchemaV2(
        bento=f"{target_schema.bento.repository.name}:{target_schema.bento.name}",
        access_type=target_schema.config.access_control,
        bentoml_config_overrides=target_schema.config.bentoml_config_overrides,
        envs=target_schema.config.envs,
    )
    if (
        target_schema.config.bento_request_overrides
        or target_schema.config.bento_function_overrides
    ):
        res.extras = ExtraDeploymentOverrides(
            bento_function_overrides=target_schema.config.bento_function_overrides,
            bento_request_overrides=target_schema.config.bento_request_overrides,
        )
    # distributed
    if target_schema.config.runners:
        svc = {}
        for name, runner in target_schema.config.runners.items():
            runner_conf = DeploymentTargetConfigV2()
            if runner.bento_function_overrides:
                runner_conf.bento_function_overrides = ApiServerBentoFunctionOverrides()
                runner_conf.bento_function_overrides.annotations = (
                    runner.bento_function_overrides.annotations
                )
                runner_conf.bento_function_overrides.extraPodMetadata = (
                    runner.bento_function_overrides.extraPodMetadata
                )
                runner_conf.bento_function_overrides.extraPodSpec = (
                    runner.bento_function_overrides.extraPodSpec
                )
            runner_conf.deployment_strategy = runner.deployment_strategy
            runner_conf.resource_instance = runner.resource_instance
            runner_conf.scaling = runner.hpa_conf
            svc[name] = runner_conf
        api_server_conf = DeploymentTargetConfigV2()
        api_server_conf.deployment_strategy = target_schema.config.deployment_strategy
        api_server_conf.bento_function_overrides = (
            target_schema.config.bento_function_overrides
        )
        api_server_conf.resource_instance = target_schema.config.resource_instance
        api_server_conf.scaling = target_schema.config.hpa_conf
        svc[target_schema.bento.repository.name] = api_server_conf
        res.services = svc

    # standalone
    else:
        res.scaling = target_schema.config.hpa_conf
        res.instance_type = target_schema.config.resource_instance
        res.deployment_strategy = target_schema.config.deployment_strategy
        res.envs = target_schema.config.envs
    return res


def fix_scaling(scaling: DeploymentTargetHPAConf | None) -> DeploymentTargetHPAConf:
    if scaling is None:
        return DeploymentTargetHPAConf(1, 1)
    if scaling.min_replicas is None:
        scaling.min_replicas = 1
    if scaling.max_replicas is None:
        scaling.max_replicas = max(scaling.min_replicas, 1)
    return scaling


@attr.define
class Deployment:
    context: t.Optional[str] = attr.field(default=None)
    cluster_name: t.Optional[str] = attr.field(default=None)
    kube_namespace: t.Optional[str] = attr.field(default=None)
    name: t.Optional[str] = attr.field(default=None)

    @classmethod
    def _get_default_kube_namespace(
        cls,
        cluster_name: str,
        context: str | None = None,
    ) -> str:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v1.get_cluster(cluster_name)
        if not res:
            raise BentoMLException("Cannot get default kube namespace")
        return res.config.default_deployment_kube_namespace

    @classmethod
    def _get_default_cluster(cls, context: str | None = None) -> str:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v1.get_cluster_list(params={"count": 1})
        if not res:
            raise BentoMLException("Failed to get list of clusters.")
        if not res.items:
            raise BentoMLException("Cannot get default clusters.")
        return res.items[0].name

    def _refetch(self) -> DeploymentSchema:
        cloud_rest_client = get_rest_api_client(self.context)
        if self.name is None:
            raise BentoMLException("Deployment name is missing")
        if self.cluster_name is None:
            self.cluster_name = self._get_default_cluster(self.context)
        if self.kube_namespace is None:
            self.kube_namespace = self._get_default_kube_namespace(
                self.cluster_name, self.context
            )
        res = cloud_rest_client.v1.get_deployment(
            self.cluster_name, self.kube_namespace, self.name
        )
        if res is None:
            raise BentoMLException(f"Deployment {self.name} is not found")
        return res

    def get_config(self) -> dict[str, t.Any]:
        schema = self._refetch()
        config = deployment_target_schema_to_v2_update_schema(schema)
        return bentoml_cattr.unstructure(config)

    @t.overload
    def update(
        self,
        project_path: str | None = ...,
        bento: Tag | str | None = ...,
        path_context: str | None = ...,
        *,
        access_type: str | None = ...,
        scaling_min: int | None = ...,
        scaling_max: int | None = ...,
        instance_type: str | None = ...,
        strategy: str | None = ...,
        envs: t.List[dict[str, t.Any]] | None = ...,
        extras: dict[str, t.Any] | None = ...,
        _bento_store: BentoStore = ...,
        _cloud_client: BentoCloudClient = ...,
    ) -> None:
        ...

    @t.overload
    def update(
        self,
        project_path: str | None = ...,
        bento: Tag | str | None = ...,
        path_context: str | None = ...,
        *,
        config_file: str | None = ...,
        _bento_store: BentoStore = ...,
        _cloud_client: BentoCloudClient = ...,
    ) -> None:
        ...

    @t.overload
    def update(
        self,
        project_path: str | None = ...,
        bento: Tag | str | None = ...,
        path_context: str | None = ...,
        *,
        config_dct: dict[str, t.Any] | None = ...,
        _bento_store: BentoStore = ...,
        _cloud_client: BentoCloudClient = ...,
    ) -> None:
        ...

    @inject
    def update(
        self,
        project_path: str | None = None,
        bento: Tag | str | None = None,
        path_context: str | None = None,
        *,
        access_type: str | None = None,
        scaling_min: int | None = None,
        scaling_max: int | None = None,
        instance_type: str | None = None,
        strategy: str | None = None,
        envs: t.List[dict[str, t.Any]] | None = None,
        extras: dict[str, t.Any] | None = None,
        config_dct: dict[str, t.Any] | None = None,
        config_file: str | None = None,
        _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> None:
        if self.name is None:
            raise BentoMLException("Deployment name is missing")

        orig_dct = self.get_config()
        cloud_rest_client = get_rest_api_client(self.context)
        if bento and project_path:
            raise BentoMLException(
                "Update a deployment needs one and only one target - either a project path or a bento"
            )
        bento_target = ""
        if project_path:
            from bentoml import bentos

            logger.info(f"Building bento: {project_path}")
            bento_target = bentos.build_bentofile(build_ctx=project_path).tag
        elif bento is not None:
            bento_target = bento

        if bento_target != "":
            bento_target = Tag.from_taglike(bento_target)
            bento_obj = _bento_store.get(bento_target)
            # try to push if bento exists, otherwise expects bentocloud to have it
            if bento_obj:
                _cloud_client.push_bento(bento=bento_obj, context=self.context)
            orig_dct["bento"] = str(bento_target)

        if config_dct:
            merging_dct = config_dct
            pass
        elif config_file:
            real_path = resolve_user_filepath(config_file, path_context)
            try:
                with open(real_path, "r") as file:
                    merging_dct = yaml.safe_load(file)
            except FileNotFoundError:
                raise ValueError(f"File not found: {real_path}")
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML file: %s", exc)
                raise
            except Exception as e:
                raise ValueError(
                    f"An error occurred while reading the file: {real_path}\n{e}"
                )
        else:
            merging_dct: dict[str, t.Any] = {"scaling": {}}
            if scaling_min is not None:
                merging_dct["scaling"]["min_replicas"] = scaling_min
            if scaling_max is not None:
                merging_dct["scaling"]["max_replicas"] = scaling_max
            if instance_type is not None:
                merging_dct["instance_type"] = instance_type

            if strategy is not None:
                merging_dct["deployment_strategy"] = strategy

            if envs is not None:
                merging_dct["envs"] = envs

            if extras is not None:
                merging_dct["extras"] = extras

            if access_type is not None:
                merging_dct["access_type"] = access_type

        config_merger.merge(orig_dct, merging_dct)
        config_struct = bentoml_cattr.structure(orig_dct, UpdateDeploymentSchemaV2)
        res = cloud_rest_client.v2.update_deployment(self.name, config_struct)
        if res is None:
            raise BentoMLException("Update deployment request failed")
        logger.debug("Deployment Schema: %s", config_struct)
        return bentoml_cattr.unstructure(res)

    def get(
        self,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> dict[str, t.Any]:
        if cluster_name is None:
            cluster_name = self._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = self._get_default_kube_namespace(cluster_name, context)
        res = self._refetch()
        self.name = deployment_name
        self.cluster_name = cluster_name
        self.context = context
        return bentoml_cattr.unstructure(res)

    def terminate(self) -> dict[str, t.Any]:
        cloud_rest_client = get_rest_api_client(self.context)
        if self.name is None:
            raise BentoMLException(
                "Deployment object needs to have a name to be terminated"
            )
        if self.cluster_name is None:
            self.cluster_name = self._get_default_cluster(self.context)
        if self.kube_namespace is None:
            self.kube_namespace = self._get_default_kube_namespace(
                self.cluster_name, self.context
            )
        res = cloud_rest_client.v1.terminate_deployment(
            self.name, self.cluster_name, self.kube_namespace
        )
        if res is None:
            raise BentoMLException(f"Deployment {self.name} is not found")
        return bentoml_cattr.unstructure(res)

    def delete(self) -> dict[str, t.Any]:
        cloud_rest_client = get_rest_api_client(self.context)
        if self.name is None:
            raise BentoMLException(
                "Deployment object needs to have a name to be deleted"
            )
        if self.cluster_name is None:
            self.cluster_name = self._get_default_cluster(self.context)
        if self.kube_namespace is None:
            self.kube_namespace = self._get_default_kube_namespace(
                self.cluster_name, self.context
            )
        res = cloud_rest_client.v1.delete_deployment(
            self.name, self.cluster_name, self.kube_namespace
        )
        if res is None:
            raise BentoMLException(f"Deployment {self.name} is not found")
        return bentoml_cattr.unstructure(res)

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
            res = cloud_rest_client.v1.get_deployment_list(cluster_name, **params)
            if res is None:
                raise BentoMLException("List deployments request failed")
            return res
        else:
            all_deployment = cloud_rest_client.v1.get_deployment_list(cluster_name)
            if all_deployment is None:
                raise BentoMLException("List deployments request failed")
            return all_deployment

    @classmethod
    @inject
    def create_deployment(
        cls,
        project_path: str | None = None,
        bento: Tag | str | None = None,
        access_type: str | None = None,
        name: str | None = None,
        cluster: str | None = None,
        scaling_min: int | None = None,
        scaling_max: int | None = None,
        instance_type: str | None = None,
        strategy: str | None = None,
        envs: t.List[dict[str, t.Any]] | None = None,
        extras: dict[str, t.Any] | None = None,
        config_dct: dict[str, t.Any] | None = None,
        config_file: str | None = None,
        path_context: str | None = None,
        context: str | None = None,
        _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> Deployment:
        cloud_rest_client = get_rest_api_client(context)
        if (not bento and not project_path) or (bento and project_path):
            raise BentoMLException(
                "Create a deployment needs one and only one target - either a project path or a bento"
            )
        bento_target = ""
        if project_path:
            from bentoml import bentos

            logger.info(f"Building bento: {project_path}")
            bento_target = bentos.build_bentofile(build_ctx=project_path).tag
        elif bento is not None:
            bento_target = bento

        bento_target = Tag.from_taglike(bento_target)
        bento_obj = _bento_store.get(bento_target)
        # try to push if bento exists, otherwise expects bentocloud to have it
        if bento_obj:
            _cloud_client.push_bento(bento=bento_obj, context=context)

        dct: dict[str, t.Any] = {"name": name, "bento": str(bento_target)}
        if config_dct:
            merging_dct = config_dct
            pass
        elif config_file:
            real_path = resolve_user_filepath(config_file, path_context)
            try:
                with open(real_path, "r") as file:
                    merging_dct = yaml.safe_load(file)
            except FileNotFoundError:
                raise ValueError(f"File not found: {real_path}")
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML file: %s", exc)
                raise
            except Exception as e:
                raise ValueError(
                    f"An error occurred while reading the file: {real_path}\n{e}"
                )
        else:
            merging_dct = {
                "scaling": {"min_replicas": scaling_min, "max_replicas": scaling_max},
                "instance_type": instance_type,
                "deployment_strategy": strategy,
                "envs": envs,
                "extras": extras,
                "access_type": access_type,
            }
        config_merger.merge(dct, merging_dct)

        # add cluster
        if "cluster" not in dct:
            cluster = cls._get_default_cluster(context)
            dct["cluster"] = cluster
        config_struct = bentoml_cattr.structure(dct, CreateDeploymentSchemaV2)

        # add scaling
        if config_struct.distributed:
            if config_struct.services is None:
                raise ValueError("The configuration for services is mandatory")
            for _, svc in config_struct.services.items():
                svc.scaling = fix_scaling(svc.scaling)
        else:
            config_struct.scaling = fix_scaling(config_struct.scaling)

        res = cloud_rest_client.v2.create_deployment(config_struct)
        if res is None:
            raise BentoMLException("Create deployment request failed")
        logger.debug("Deployment Schema: %s", config_struct)
        return Deployment(
            context=context,
            cluster_name=res.cluster.name,
            kube_namespace=res.kube_namespace,
        )

    @classmethod
    @inject
    def update_deployment(
        cls,
        name: str,
        project_path: str | None = None,
        bento: Tag | str | None = None,
        access_type: str | None = None,
        cluster: str | None = None,
        scaling_min: int | None = None,
        scaling_max: int | None = None,
        instance_type: str | None = None,
        strategy: str | None = None,
        envs: t.List[dict[str, t.Any]] | None = None,
        extras: dict[str, t.Any] | None = None,
        config_dct: dict[str, t.Any] | None = None,
        config_file: str | None = None,
        path_context: str | None = None,
        context: str | None = None,
        _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> Deployment:
        deployment = Deployment(name=name, context=context, cluster_name=cluster)
        if config_dct:
            deployment.update(
                project_path=project_path,
                bento=bento,
                config_dct=config_dct,
                _bento_store=_bento_store,
                _cloud_client=_cloud_client,
                path_context=path_context,
            )
        elif config_file:
            deployment.update(
                project_path=project_path,
                bento=bento,
                config_file=config_file,
                _bento_store=_bento_store,
                _cloud_client=_cloud_client,
                path_context=path_context,
            )
        else:
            deployment.update(
                project_path=project_path,
                bento=bento,
                access_type=access_type,
                scaling_min=scaling_min,
                scaling_max=scaling_max,
                instance_type=instance_type,
                strategy=strategy,
                envs=envs,
                extras=extras,
                _bento_store=_bento_store,
                _cloud_client=_cloud_client,
                path_context=path_context,
            )
        return deployment

    @classmethod
    def get_deployment(
        cls,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> Deployment:
        dep = Deployment(context=context)
        dep.get(
            deployment_name=deployment_name,
            context=context,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
        )
        return dep

    @classmethod
    def terminate_deployment(
        cls,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> Deployment:
        dep = Deployment(
            context=context,
            name=deployment_name,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
        )
        dep.terminate()
        return dep

    @classmethod
    def delete_deployment(
        cls,
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> Deployment:
        dep = Deployment(
            context=context,
            name=deployment_name,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
        )
        dep.terminate()
        return dep
