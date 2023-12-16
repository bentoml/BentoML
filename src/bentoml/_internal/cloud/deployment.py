from __future__ import annotations

import logging
import time
import typing as t

import attr
import yaml
from deepmerge.merger import Merger
from simple_di import Provide
from simple_di import inject

if t.TYPE_CHECKING:
    from bentoml_io.client import HTTPClient

    from bentoml._internal.bento.bento import BentoStore
    from bentoml._internal.cloud.bentocloud import BentoCloudClient

    from .schemas.schemasv1 import DeploymentFullSchema
    from .schemas.schemasv1 import DeploymentSchema

from ...exceptions import BentoMLException
from ...exceptions import NotFound
from ..configuration.containers import BentoMLContainer
from ..tag import Tag
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from .config import get_rest_api_client
from .schemas.modelschemas import ApiServerBentoFunctionOverrides
from .schemas.modelschemas import DeploymentStatus
from .schemas.modelschemas import DeploymentTargetConfigV2
from .schemas.modelschemas import DeploymentTargetHPAConf
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


def convert_deployment_target_schema_to_v2_update_schema(
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
    _schema: t.Optional[DeploymentFullSchema] = attr.field(
        alias="_schema", default=None
    )
    _urls: t.Optional[list[str]] = attr.field(alias="_urls", default=None)

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

    def _refetch(self) -> None:
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
            raise NotFound(f"Deployment {self.name} is not found")
        self._schema = res
        self._urls = res.urls

    def _get_update_schema(self) -> dict[str, t.Any]:
        if self._schema is None:
            raise BentoMLException("schema is empty")
        config = convert_deployment_target_schema_to_v2_update_schema(self._schema)
        return bentoml_cattr.unstructure(config)

    def get_config(self) -> dict[str, t.Any]:
        self._refetch()
        res = self._get_update_schema()
        # bento should not be in the deployment config
        del res["bento"]
        return res

    def get_bento(self) -> str:
        self._refetch()
        res = self._get_update_schema()
        return res["bento"]

    def get_status(self) -> str:
        self._refetch()
        if self._schema is None:
            raise BentoMLException("schema is empty")
        return str(self._schema.status)

    def get_client(
        self,
        url: str,
        is_async: bool = False,
        media_type: str = "application/json",
        token: str | None = None,
    ) -> HTTPClient:
        from bentoml_io.client import AsyncHTTPClient
        from bentoml_io.client import SyncHTTPClient

        self._refetch()
        if self._schema is None:
            raise BentoMLException("schema is empty")
        if self._schema.status != DeploymentStatus.Running:
            raise BentoMLException(f"Deployment status is {self._schema.status}")
        if len(self._schema.urls) != 1:
            raise BentoMLException("Deployment url is not ready")
        if is_async:
            return AsyncHTTPClient(url, media_type=media_type, token=token)
        else:
            return SyncHTTPClient(url, media_type=media_type, token=token)

    def wait_until_ready(self, timeout: int = 300, check_interval: int = 5) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            if status == "running":
                logger.info(f"Deployment '{self.name}' is ready.")
                return
            logger.info(
                f"Waiting for deployment '{self.name}' to be ready. Current status: '{status}'."
            )
            time.sleep(check_interval)

        raise TimeoutError(
            f"Timed out waiting for deployment '{self.name}' to be ready."
        )

    @classmethod
    def list(
        cls,
        context: str | None = None,
        cluster_name: str | None = None,
        search: str | None = None,
    ) -> list[Deployment]:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            res_count = cloud_rest_client.v1.get_organization_deployment_list(
                search=search
            )
            if res_count is None:
                raise BentoMLException("List deployments request failed")
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v1.get_organization_deployment_list(
                search=search, count=res_count.total
            )
            if res is None:
                raise BentoMLException("List deployments request failed")
        else:
            res_count = cloud_rest_client.v1.get_cluster_deployment_list(
                cluster_name, search=search
            )
            if res_count is None:
                raise NotFound(f"Cluster {cluster_name} is not found")
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v1.get_cluster_deployment_list(
                cluster_name, search=search, count=res_count.total
            )
            if res is None:
                raise BentoMLException("List deployments request failed")
        return [
            Deployment(
                name=schema.name,
                context=context,
                kube_namespace=schema.kube_namespace,
                cluster_name=schema.cluster.name,
            )
            for schema in res.items
        ]

    @classmethod
    @inject
    def create(
        cls,
        project_path: str | None = None,
        bento: Tag | str | None = None,
        access_type: str | None = None,
        name: str | None = None,
        cluster_name: str | None = None,
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
        if bento and project_path:
            raise BentoMLException(
                "Create a deployment needs only one target - either a project path or a bento"
            )
        if project_path:
            from bentoml import bentos

            logger.info(f"Building bento: {project_path}")
            bento = bentos.build_bentofile(
                build_ctx=project_path, _bento_store=_bento_store
            ).tag

        if bento is None:
            raise BentoMLException(
                "Create a deployment needs a target; project path or bento is necessary"
            )
        bento = Tag.from_taglike(bento)
        try:
            bento_obj = _bento_store.get(bento)
        except NotFound as e:
            # "bento repo needs to exist if it is latest"
            if bento.version is None or bento.version == "latest":
                raise e
            bento_obj = None

        # try to push if bento exists, otherwise expects bentocloud to have it
        if bento_obj:
            _cloud_client.push_bento(bento=bento_obj, context=context)
            bento = bento_obj.tag

        dct: dict[str, t.Any] = {"name": name, "bento": str(bento)}
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
            cluster_name = cls._get_default_cluster(context)
            dct["cluster"] = cluster_name
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
            _schema=res,
        )

    @classmethod
    @inject
    def update(
        cls,
        name: str,
        project_path: str | None = None,
        bento: Tag | str | None = None,
        access_type: str | None = None,
        cluster_name: str | None = None,
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
        deployment = Deployment.get(
            name=name, context=context, cluster_name=cluster_name
        )
        orig_dct = deployment._get_update_schema()
        cloud_rest_client = get_rest_api_client(context)
        if bento and project_path:
            raise BentoMLException(
                "Update a deployment needs one and only one target - either a project path or a bento"
            )
        if project_path:
            from bentoml import bentos

            logger.info(f"Building bento: {project_path}")
            bento = bentos.build_bentofile(build_ctx=project_path).tag

        if bento is not None:
            bento = Tag.from_taglike(bento)
            try:
                bento_obj = _bento_store.get(bento)
            except NotFound as e:
                # "bento repo needs to exist if it is latest"
                if bento.version is None or bento.version == "latest":
                    raise e
                bento_obj = None

            # try to push if bento exists, otherwise expects bentocloud to have it
            if bento_obj:
                _cloud_client.push_bento(bento=bento_obj, context=context)
                bento = bento_obj.tag

            orig_dct["bento"] = str(bento)

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
        res = cloud_rest_client.v2.update_deployment(name, config_struct)
        if res is None:
            raise BentoMLException("Update deployment request failed")
        logger.debug("Deployment Schema: %s", config_struct)
        deployment._schema = res
        deployment._urls = res.urls
        return deployment

    @classmethod
    @inject
    def apply(
        cls,
        name: str,
        project_path: str | None = None,
        bento: Tag | str | None = None,
        cluster_name: str | None = None,
        config_dct: dict[str, t.Any] | None = None,
        config_file: str | None = None,
        path_context: str | None = None,
        context: str | None = None,
        _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> Deployment:
        try:
            deployment = Deployment.get(
                name=name, context=context, cluster_name=cluster_name
            )
        except NotFound:
            return Deployment.create(
                name=name,
                project_path=project_path,
                bento=bento,
                cluster_name=cluster_name,
                config_dct=config_dct,
                config_file=config_file,
                context=context,
                path_context=path_context,
                _bento_store=_bento_store,
                _cloud_client=_cloud_client,
            )
        cloud_rest_client = get_rest_api_client(context)

        if bento and project_path:
            raise BentoMLException(
                "Update a deployment needs one and only one target - either a project path or a bento"
            )

        if project_path:
            from bentoml import bentos

            logger.info(f"Building bento: {project_path}")
            bento = bentos.build_bentofile(build_ctx=project_path).tag

        if bento is not None:
            bento = Tag.from_taglike(bento)
            try:
                bento_obj = _bento_store.get(bento)
            except NotFound as e:
                # "bento repo needs to exist if it is latest"
                if bento.version is None or bento.version == "latest":
                    raise e
                bento_obj = None

            # try to push if bento exists, otherwise expects bentocloud to have it
            if bento_obj:
                _cloud_client.push_bento(bento=bento_obj, context=context)
                bento = bento_obj.tag
        else:
            bento = deployment._get_update_schema()["bento"]

        schema_dct: dict[str, t.Any] = {"bento": str(bento)}

        if config_file:
            real_path = resolve_user_filepath(config_file, path_context)
            try:
                with open(real_path, "r") as file:
                    config_dct = yaml.safe_load(file)
            except FileNotFoundError:
                raise ValueError(f"File not found: {real_path}")
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML file: %s", exc)
                raise
            except Exception as e:
                raise ValueError(
                    f"An error occurred while reading the file: {real_path}\n{e}"
                )
        if config_dct is None:
            raise BentoMLException("Apply a deployment needs a configuration input")

        schema_dct.update(config_dct)
        config_struct = bentoml_cattr.structure(schema_dct, UpdateDeploymentSchemaV2)
        res = cloud_rest_client.v2.update_deployment(name, config_struct)
        if res is None:
            raise BentoMLException("Apply deployment request failed")
        logger.debug("Deployment Schema: %s", config_struct)
        deployment._schema = res
        deployment._urls = res.urls
        return deployment

    @classmethod
    def get(
        cls,
        name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> Deployment:
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)
        deployment = Deployment(
            context=context,
            cluster_name=cluster_name,
            name=name,
            kube_namespace=kube_namespace,
        )
        deployment._refetch()
        return deployment

    @classmethod
    def terminate(
        cls,
        name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> Deployment:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)
        res = cloud_rest_client.v1.terminate_deployment(
            cluster_name, kube_namespace, name
        )
        if res is None:
            raise NotFound(f"Deployment {name} is not found")
        return Deployment(
            name=name,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
            context=context,
            _schema=res,
        )

    @classmethod
    def delete(
        cls,
        name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> None:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        if kube_namespace is None:
            kube_namespace = cls._get_default_kube_namespace(cluster_name, context)
        res = cloud_rest_client.v1.delete_deployment(cluster_name, kube_namespace, name)
        if res is None:
            raise BentoMLException("Update deployment request failed")
