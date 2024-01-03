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
    from _bentoml_impl.client import AsyncHTTPClient
    from _bentoml_impl.client import SyncHTTPClient
    from bentoml._internal.bento.bento import BentoStore
    from bentoml._internal.cloud.bentocloud import BentoCloudClient


from ...exceptions import BentoMLException
from ...exceptions import NotFound
from ..configuration.containers import BentoMLContainer
from ..tag import Tag
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from .config import get_rest_api_client
from .schemas.modelschemas import AccessControl
from .schemas.modelschemas import DeploymentStatus
from .schemas.modelschemas import DeploymentTargetHPAConf
from .schemas.schemasv2 import CreateDeploymentSchema as CreateDeploymentSchemaV2
from .schemas.schemasv2 import DeploymentSchema
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


@inject
def get_real_bento_tag(
    project_path: str | None = None,
    bento: str | Tag | None = None,
    context: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Tag:
    if project_path:
        from bentoml.bentos import build_bentofile

        bento_obj = build_bentofile(build_ctx=project_path, _bento_store=_bento_store)
        _cloud_client.push_bento(bento=bento_obj, context=context)
        return bento_obj.tag
    elif bento:
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
        return bento
    else:
        raise BentoMLException(
            "Create a deployment needs a target; project path or bento is necessary"
        )


@attr.define
class DeploymentInfo:
    __omit_if_default__ = True
    name: str
    created_at: str
    bento: Tag
    status: DeploymentStatus
    admin_console: str
    endpoint: t.Optional[str]
    config: dict[str, t.Any]

    def to_dict(self) -> t.Dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)


@attr.define
class Deployment:
    context: t.Optional[str]
    cluster_name: str
    name: str
    _schema: DeploymentSchema = attr.field(alias="_schema", repr=False)
    _urls: t.Optional[list[str]] = attr.field(alias="_urls", default=None)

    @staticmethod
    def _fix_scaling(
        scaling: DeploymentTargetHPAConf | None,
    ) -> DeploymentTargetHPAConf:
        if scaling is None:
            return DeploymentTargetHPAConf(1, 1)
        if scaling.min_replicas is None:
            scaling.min_replicas = 1
        if scaling.max_replicas is None:
            scaling.max_replicas = max(scaling.min_replicas, 1)
        # one edge case:
        if scaling.min_replicas > scaling.max_replicas:
            scaling.min_replicas = scaling.max_replicas
            logger.warning(
                "min scaling value is greater than max scaling value, setting min scaling to max scaling value"
            )
        if scaling.min_replicas < 0:
            raise BentoMLException(
                "min scaling values must be greater than or equal to 0"
            )
        if scaling.max_replicas <= 0:
            raise BentoMLException("max scaling values must be greater than 0")
        return scaling

    @staticmethod
    def _validate_input_on_distributed(
        config_struct: UpdateDeploymentSchemaV2, distributed: bool
    ) -> None:
        if distributed:
            if config_struct.instance_type is not None:
                raise BentoMLException(
                    "The 'instance_type' field is not allowed for distributed deployments. Please specify it per service in the services field."
                )
            if (
                config_struct.scaling is not None
                and config_struct.scaling != DeploymentTargetHPAConf()
            ):
                raise BentoMLException(
                    "The 'scaling' field is not allowed for distributed deployments. Please specify it per service in the services field."
                )
            if config_struct.deployment_strategy is not None:
                raise BentoMLException(
                    "The 'deployment_strategy' field is not allowed for distributed deployments. Please specify it per service in the services field."
                )
            if config_struct.extras is not None:
                raise BentoMLException(
                    "The 'extras' field is not allowed for distributed deployments. Please specify it per service in the services field."
                )
            if config_struct.cold_start_timeout is not None:
                raise BentoMLException(
                    "The 'cold_start_timeout' field is not allowed for distributed deployments. Please specify it per service in the services field."
                )
        elif not distributed:
            if config_struct.services != {}:
                raise BentoMLException(
                    "The 'services' field is only allowed for distributed deployments."
                )

    @classmethod
    def _fix_and_validate_schema(
        cls, config_struct: UpdateDeploymentSchemaV2, distributed: bool
    ):
        cls._validate_input_on_distributed(config_struct, distributed)
        # fix scaling
        if distributed:
            if len(config_struct.services) == 0:
                raise BentoMLException("The configuration for services is mandatory")
            for _, svc in config_struct.services.items():
                svc.scaling = cls._fix_scaling(svc.scaling)
        else:
            config_struct.scaling = cls._fix_scaling(config_struct.scaling)
        if config_struct.access_type is None:
            config_struct.access_type = AccessControl.PUBLIC

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
        res = cloud_rest_client.v2.get_deployment(self.cluster_name, self.name)
        if res is None:
            raise NotFound(f"deployment {self.name} is not found")
        self._schema = res
        self._urls = res.urls

    def _conver_schema_to_update_schema(self) -> dict[str, t.Any]:
        if self._schema.latest_revision is None:
            raise BentoMLException(
                f"Deployment {self._schema.name} has no latest revision"
            )
        target_schema = self._schema.latest_revision.targets[0]
        if target_schema is None:
            raise BentoMLException(f"Deployment {self._schema.name} has no target")
        if target_schema.config is None:
            raise BentoMLException(f"Deployment {self._schema.name} has no config")
        if target_schema.bento is None:
            raise BentoMLException(f"Deployment {self._schema.name} has no bento")
        update_schema = UpdateDeploymentSchemaV2(
            services=target_schema.config.services,
            instance_type=target_schema.config.instance_type,
            deployment_strategy=target_schema.config.deployment_strategy,
            scaling=target_schema.config.scaling,
            envs=target_schema.config.envs,
            extras=target_schema.config.extras,
            access_type=target_schema.config.access_type,
            bentoml_config_overrides=target_schema.config.bentoml_config_overrides,
            bento=target_schema.bento.repository.name + ":" + target_schema.bento.name,
            cold_start_timeout=target_schema.config.cold_start_timeout,
        )
        return bentoml_cattr.unstructure(update_schema)

    def _conver_schema_to_bento(self) -> Tag:
        if self._schema.latest_revision is None:
            raise BentoMLException(
                f"Deployment {self._schema.name} has no latest revision"
            )
        target_schema = self._schema.latest_revision.targets[0]
        if target_schema is None:
            raise BentoMLException(f"Deployment {self._schema.name} has no target")
        if target_schema.bento is None:
            raise BentoMLException(f"Deployment {self._schema.name} has no bento")
        return Tag.from_taglike(
            target_schema.bento.repository.name + ":" + target_schema.bento.name
        )

    @property
    def info(self) -> DeploymentInfo:
        schema = self._conver_schema_to_update_schema()
        del schema["bento"]
        return DeploymentInfo(
            name=self.name,
            bento=self._conver_schema_to_bento(),
            status=self._schema.status,
            admin_console=self.get_bento_cloud_url(),
            endpoint=self._urls[0] if self._urls else None,
            config=schema,
            created_at=self._schema.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def get_config(self) -> dict[str, t.Any]:
        self._refetch()
        res = self._conver_schema_to_update_schema()
        # bento should not be in the deployment config
        del res["bento"]
        return res

    def get_bento(self) -> str:
        self._refetch()
        return str(self._conver_schema_to_bento())

    def get_status(self) -> str:
        self._refetch()
        return self._schema.status.value

    def get_client(
        self,
        is_async: bool = False,
        media_type: str = "application/json",
        token: str | None = None,
    ) -> SyncHTTPClient:
        from _bentoml_impl.client import SyncHTTPClient

        self._refetch()
        if self._schema.status != DeploymentStatus.Running:
            raise BentoMLException(f"Deployment status is {self._schema.status}")
        if self._urls is None or len(self._urls) != 1:
            raise BentoMLException("Deployment url is not ready")
        return SyncHTTPClient(self._urls[0], media_type=media_type, token=token)

    def get_bento_cloud_url(self) -> str:
        client = get_rest_api_client(self.context)
        namespace = self._get_default_kube_namespace(self.cluster_name, self.context)
        return f"{client.v1.endpoint}/clusters/{self.cluster_name}/namespaces/{namespace}/deployments/{self.name}"

    def get_async_client(
        self,
        media_type: str = "application/json",
        token: str | None = None,
    ) -> AsyncHTTPClient:
        from _bentoml_impl.client import AsyncHTTPClient

        self._refetch()
        if self._schema.status != DeploymentStatus.Running:
            raise BentoMLException(f"Deployment status is {self._schema.status}")
        if self._urls is None or len(self._urls) != 1:
            raise BentoMLException("Deployment url is not ready")
        return AsyncHTTPClient(self._urls[0], media_type=media_type, token=token)

    def wait_until_ready(self, timeout: int = 300, check_interval: int = 5) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            if status == DeploymentStatus.Running.value:
                logger.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Deployment '{self.name}' is ready."
                )
                return
            logger.info(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for deployment '{self.name}' to be ready. Current status: '{status}'."
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
            res_count = cloud_rest_client.v2.list_deployment(all=True, search=search)
            if res_count is None:
                raise BentoMLException("List deployments request failed")
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v2.list_deployment(
                search=search, count=res_count.total, all=True
            )
            if res is None:
                raise BentoMLException("List deployments request failed")
        else:
            res_count = cloud_rest_client.v2.list_deployment(
                cluster_name, search=search
            )
            if res_count is None:
                raise NotFound(f"Cluster {cluster_name} is not found")
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v2.list_deployment(
                cluster_name, search=search, count=res_count.total
            )
            if res is None:
                raise BentoMLException("List deployments request failed")
        return [
            Deployment(
                name=schema.name,
                context=context,
                cluster_name=schema.cluster.name,
                _schema=schema,
            )
            for schema in res.items
        ]

    @classmethod
    def create(
        cls,
        bento: Tag,
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
        config_file: str | t.TextIO | None = None,
        path_context: str | None = None,
        context: str | None = None,
    ) -> Deployment:
        cloud_rest_client = get_rest_api_client(context)
        dct: dict[str, t.Any] = {
            "bento": str(bento),
        }
        if name:
            dct["name"] = name
        else:
            # the cloud takes care of the name
            dct["name"] = ""

        if config_dct:
            merging_dct = config_dct
            pass
        elif isinstance(config_file, str):
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
        elif config_file is not None:
            try:
                merging_dct = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML config-file stream: %s", exc)
                raise
        else:
            merging_dct = {
                "scaling": {"min_replicas": scaling_min, "max_replicas": scaling_max},
                "instance_type": instance_type,
                "deployment_strategy": strategy,
                "envs": envs,
                "extras": extras,
                "access_type": access_type,
                "cluster": cluster_name,
            }
        dct.update(merging_dct)

        # add cluster
        if "cluster" not in dct or dct["cluster"] is None:
            cluster_name = cls._get_default_cluster(context)
            dct["cluster"] = cluster_name

        if "distributed" not in dct:
            dct["distributed"] = (
                "services" in dct
                and dct["services"] is not None
                and dct["services"] != {}
            )

        config_struct = bentoml_cattr.structure(dct, CreateDeploymentSchemaV2)
        cls._fix_and_validate_schema(config_struct, dct["distributed"])

        res = cloud_rest_client.v2.create_deployment(
            create_schema=config_struct, cluster_name=config_struct.cluster
        )
        logger.debug("Deployment Schema: %s", config_struct)
        return Deployment(
            context=context,
            cluster_name=config_struct.cluster,
            name=res.name,
            _schema=res,
        )

    @classmethod
    def update(
        cls,
        name: str,
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
        config_file: str | t.TextIO | None = None,
        path_context: str | None = None,
        context: str | None = None,
    ) -> Deployment:
        deployment = Deployment.get(
            name=name, context=context, cluster_name=cluster_name
        )
        orig_dct = deployment._conver_schema_to_update_schema()
        distributed = deployment._schema.distributed
        cloud_rest_client = get_rest_api_client(context)
        if bento:
            orig_dct["bento"] = str(bento)

        if config_dct:
            merging_dct = config_dct
            pass
        elif isinstance(config_file, str):
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
        elif config_file is not None:
            try:
                merging_dct = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML config-file stream: %s", exc)
                raise

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

        cls._fix_and_validate_schema(config_struct, distributed)

        res = cloud_rest_client.v2.update_deployment(
            cluster_name=deployment.cluster_name,
            deployment_name=name,
            update_schema=config_struct,
        )
        if res is None:
            raise NotFound(f"deployment {name} is not found")
        logger.debug("Deployment Schema: %s", config_struct)
        deployment._schema = res
        deployment._urls = res.urls
        return deployment

    @classmethod
    def apply(
        cls,
        name: str,
        bento: Tag | None = None,
        cluster_name: str | None = None,
        config_dct: dict[str, t.Any] | None = None,
        config_file: str | None = None,
        path_context: str | None = None,
        context: str | None = None,
    ) -> Deployment:
        try:
            deployment = Deployment.get(
                name=name, context=context, cluster_name=cluster_name
            )
        except NotFound as e:
            if bento is not None:
                return cls.create(
                    bento=bento,
                    name=name,
                    cluster_name=cluster_name,
                    config_dct=config_dct,
                    config_file=config_file,
                    path_context=path_context,
                    context=context,
                )
            else:
                raise e
        cloud_rest_client = get_rest_api_client(context)
        if bento is None:
            bento = deployment._conver_schema_to_bento()

        schema_dct: dict[str, t.Any] = {"bento": str(bento)}
        distributed = deployment._schema.distributed

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
        cls._fix_and_validate_schema(config_struct, distributed)

        res = cloud_rest_client.v2.update_deployment(
            deployment_name=name,
            update_schema=config_struct,
            cluster_name=deployment.cluster_name,
        )
        if res is None:
            raise NotFound(f"deployment {name} is not found")
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
    ) -> Deployment:
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.get_deployment(cluster_name, name)
        if res is None:
            raise NotFound(f"deployment {name} is not found")

        deployment = Deployment(
            context=context,
            cluster_name=cluster_name,
            name=name,
            _schema=res,
            _urls=res.urls,
        )
        return deployment

    @classmethod
    def terminate(
        cls,
        name: str,
        context: str | None = None,
        cluster_name: str | None = None,
    ) -> Deployment:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        res = cloud_rest_client.v2.terminate_deployment(cluster_name, name)
        if res is None:
            raise NotFound(f"Deployment {name} is not found")
        return Deployment(
            name=name,
            cluster_name=cluster_name,
            context=context,
            _schema=res,
            _urls=res.urls,
        )

    @classmethod
    def delete(
        cls,
        name: str,
        context: str | None = None,
        cluster_name: str | None = None,
    ) -> None:
        cloud_rest_client = get_rest_api_client(context)
        if cluster_name is None:
            cluster_name = cls._get_default_cluster(context)
        res = cloud_rest_client.v2.delete_deployment(cluster_name, name)
        if res is None:
            raise NotFound(f"Deployment {name} is not found")
