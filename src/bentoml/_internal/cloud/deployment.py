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
from .schemas.schemasv2 import DeploymentTargetSchema
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


def get_args_from_config(
    name: str | None = None,
    bento: Tag | str | None = None,
    cluster: str | None = None,
    config_dict: dict[str, t.Any] | None = None,
    config_file: str | t.TextIO | None = None,
    path_context: str | None = None,
):
    file_dict: dict[str, t.Any] | None = None

    if name is None and config_dict is not None and "name" in config_dict:
        name = config_dict["name"]
    if bento is None and config_dict is not None and "bento" in config_dict:
        bento = config_dict["bento"]
    if cluster is None and config_dict is not None and "cluster" in config_dict:
        cluster = config_dict["cluster"]

    if isinstance(config_file, str):
        real_path = resolve_user_filepath(config_file, path_context)
        try:
            with open(real_path, "r") as file:
                file_dict = yaml.safe_load(file)
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
            file_dict = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            logger.error("Error while parsing YAML config-file stream: %s", exc)
            raise

    if file_dict is not None:
        if bento is None and "bento" in file_dict:
            bento = file_dict["bento"]
        if name is None and "name" in file_dict:
            name = file_dict["name"]
        if cluster is None and "cluster" in file_dict:
            cluster = file_dict["cluster"]

    return name, bento, cluster


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
class DeploymentConfig(CreateDeploymentSchemaV2):
    def to_yaml(self):
        return yaml.dump(bentoml_cattr.unstructure(self))

    def to_dict(self):
        return bentoml_cattr.unstructure(self)


@attr.define
class DeploymentState:
    status: str
    created_at: str
    updated_at: str
    # no error message for now
    # error_msg: str

    def to_dict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)


@attr.define
class DeploymentInfo:
    name: str
    admin_console: str
    created_at: str
    created_by: str
    cluster: str
    organization: str
    distributed: bool
    description: t.Optional[str]
    _context: t.Optional[str] = attr.field(alias="_context", repr=False)
    _schema: DeploymentSchema = attr.field(alias="_schema", repr=False)
    _urls: t.Optional[list[str]] = attr.field(alias="_urls", default=None, repr=False)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "cluster": self.cluster,
            "description": self.description,
            "organization": self.organization,
            "admin_console": self.admin_console,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "distributed": self.distributed,
            "config": self.get_config(refetch=False).to_dict(),
            "status": self.get_status(refetch=False).to_dict(),
        }

    def _refetch(self) -> None:
        res = Deployment.get(self.name, self.cluster, self._context)
        self._schema = res._schema
        self._urls = res._urls

    def _refetch_target(self, refetch: bool) -> DeploymentTargetSchema:
        if refetch:
            self._refetch()
        if self._schema.latest_revision is None:
            raise BentoMLException(f"Deployment {self.name} has no latest revision")
        if len(self._schema.latest_revision.targets) == 0:
            raise BentoMLException(
                f"Deployment {self.name} has no latest revision targets"
            )
        target = self._schema.latest_revision.targets[0]
        if target is None:
            raise BentoMLException(f"Deployment {self.name} has no target")
        return target

    def get_config(self, refetch: bool = True) -> DeploymentConfig:
        target = self._refetch_target(refetch)
        if target.config is None:
            raise BentoMLException(f"Deployment {self.name} has no config")

        return DeploymentConfig(
            name=self.name,
            bento=self.get_bento(refetch=False),
            distributed=self.distributed,
            description=self.description,
            services=target.config.services,
            instance_type=target.config.instance_type,
            deployment_strategy=target.config.deployment_strategy,
            scaling=target.config.scaling,
            envs=target.config.envs,
            extras=target.config.extras,
            access_type=target.config.access_type,
            bentoml_config_overrides=target.config.bentoml_config_overrides,
            cold_start_timeout=target.config.cold_start_timeout,
        )

    def get_status(self, refetch: bool = True) -> DeploymentState:
        if refetch:
            self._refetch()
        updated_at = self._schema.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if self._schema.updated_at is not None:
            updated_at = self._schema.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        return DeploymentState(
            status=self._schema.status.value,
            created_at=self._schema.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=updated_at,
        )

    def get_bento(self, refetch: bool = True) -> str:
        target = self._refetch_target(refetch)
        if target.bento is None:
            raise BentoMLException(f"Deployment {self.name} has no bento")
        return target.bento.repository.name + ":" + target.bento.version

    def get_client(
        self,
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

    def wait_until_ready(self, timeout: int = 3600, check_interval: int = 30) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status()
            if status.status == DeploymentStatus.Running.value:
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


@attr.define
class Deployment:
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

    @staticmethod
    def _convert_schema_to_update_schema(_schema: DeploymentSchema) -> dict[str, t.Any]:
        if _schema.latest_revision is None:
            raise BentoMLException(f"Deployment {_schema.name} has no latest revision")
        if len(_schema.latest_revision.targets) == 0:
            raise BentoMLException(
                f"Deployment {_schema.name} has no latest revision targets"
            )
        target_schema = _schema.latest_revision.targets[0]
        if target_schema is None:
            raise BentoMLException(f"Deployment {_schema.name} has no target")
        if target_schema.config is None:
            raise BentoMLException(f"Deployment {_schema.name} has no config")
        if target_schema.bento is None:
            raise BentoMLException(f"Deployment {_schema.name} has no bento")
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

    @staticmethod
    def _convert_schema_to_bento(_schema: DeploymentSchema) -> Tag:
        if _schema.latest_revision is None:
            raise BentoMLException(f"Deployment {_schema.name} has no latest revision")
        target_schema = _schema.latest_revision.targets[0]
        if target_schema is None:
            raise BentoMLException(f"Deployment {_schema.name} has no target")
        if target_schema.bento is None:
            raise BentoMLException(f"Deployment {_schema.name} has no bento")
        return Tag.from_taglike(
            target_schema.bento.repository.name + ":" + target_schema.bento.name
        )

    @staticmethod
    def _generate_deployment_info_(
        context: str | None, res: DeploymentSchema, urls: list[str] | None = None
    ) -> DeploymentInfo:
        client = get_rest_api_client(context)
        cluster_display_name = res.cluster.host_cluster_display_name
        if cluster_display_name is None:
            cluster_display_name = res.cluster.name
        return DeploymentInfo(
            name=res.name,
            # TODO: update this after the url in the frontend is fixed
            admin_console=f"{client.v1.endpoint}/clusters/{res.cluster.name}/namespaces/{res.kube_namespace}/deployments/{res.name}",
            created_at=res.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            created_by=res.creator.name,
            cluster=cluster_display_name,
            organization=res.cluster.organization_name,
            distributed=res.distributed,
            description=res.description,
            _schema=res,
            _context=context,
            _urls=urls,
        )

    @classmethod
    def list(
        cls,
        context: str | None = None,
        cluster: str | None = None,
        search: str | None = None,
    ) -> list[DeploymentInfo]:
        cloud_rest_client = get_rest_api_client(context)
        if cluster is None:
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
            res_count = cloud_rest_client.v2.list_deployment(cluster, search=search)
            if res_count is None:
                raise NotFound(f"Cluster {cluster} is not found")
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v2.list_deployment(
                cluster, search=search, count=res_count.total
            )
            if res is None:
                raise BentoMLException("List deployments request failed")
        return [cls._generate_deployment_info_(context, schema) for schema in res.items]

    @classmethod
    def create(
        cls,
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
        config_dict: dict[str, t.Any] | None = None,
        config_file: str | t.TextIO | None = None,
        path_context: str | None = None,
        context: str | None = None,
    ) -> DeploymentInfo:
        cloud_rest_client = get_rest_api_client(context)
        dict: dict[str, t.Any] = {
            "bento": str(bento),
        }
        if name:
            dict["name"] = name
        else:
            # the cloud takes care of the name
            dict["name"] = ""

        if config_dict:
            merging_dict = config_dict
            pass
        elif isinstance(config_file, str):
            real_path = resolve_user_filepath(config_file, path_context)
            try:
                with open(real_path, "r") as file:
                    merging_dict = yaml.safe_load(file)
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
                merging_dict = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML config-file stream: %s", exc)
                raise
        else:
            merging_dict = {
                "scaling": {"min_replicas": scaling_min, "max_replicas": scaling_max},
                "instance_type": instance_type,
                "deployment_strategy": strategy,
                "envs": envs,
                "extras": extras,
                "access_type": access_type,
            }
        dict.update(merging_dict)

        if "distributed" not in dict:
            dict["distributed"] = (
                "services" in dict
                and dict["services"] is not None
                and dict["services"] != {}
            )

        config_struct = bentoml_cattr.structure(dict, CreateDeploymentSchemaV2)
        cls._fix_and_validate_schema(config_struct, dict["distributed"])

        res = cloud_rest_client.v2.create_deployment(
            create_schema=config_struct, cluster=cluster
        )

        logger.debug("Deployment Schema: %s", config_struct)

        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def update(
        cls,
        name: str | None,
        bento: Tag | str | None = None,
        access_type: str | None = None,
        cluster: str | None = None,
        scaling_min: int | None = None,
        scaling_max: int | None = None,
        instance_type: str | None = None,
        strategy: str | None = None,
        envs: t.List[dict[str, t.Any]] | None = None,
        extras: dict[str, t.Any] | None = None,
        config_dict: dict[str, t.Any] | None = None,
        config_file: str | t.TextIO | None = None,
        path_context: str | None = None,
        context: str | None = None,
    ) -> DeploymentInfo:
        if name is None:
            raise ValueError("name is required")
        cloud_rest_client = get_rest_api_client(context)
        deployment_schema = cloud_rest_client.v2.get_deployment(name, cluster)
        if deployment_schema is None:
            raise NotFound(f"deployment {name} is not found")

        orig_dict = cls._convert_schema_to_update_schema(deployment_schema)
        distributed = deployment_schema.distributed
        if bento:
            orig_dict["bento"] = str(bento)

        if config_dict:
            merging_dict = config_dict
            pass
        elif isinstance(config_file, str):
            real_path = resolve_user_filepath(config_file, path_context)
            try:
                with open(real_path, "r") as file:
                    merging_dict = yaml.safe_load(file)
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
                merging_dict = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML config-file stream: %s", exc)
                raise

        else:
            merging_dict: dict[str, t.Any] = {"scaling": {}}
            if scaling_min is not None:
                merging_dict["scaling"]["min_replicas"] = scaling_min
            if scaling_max is not None:
                merging_dict["scaling"]["max_replicas"] = scaling_max
            if instance_type is not None:
                merging_dict["instance_type"] = instance_type

            if strategy is not None:
                merging_dict["deployment_strategy"] = strategy

            if envs is not None:
                merging_dict["envs"] = envs

            if extras is not None:
                merging_dict["extras"] = extras

            if access_type is not None:
                merging_dict["access_type"] = access_type

        config_merger.merge(orig_dict, merging_dict)

        config_struct = bentoml_cattr.structure(orig_dict, UpdateDeploymentSchemaV2)

        cls._fix_and_validate_schema(config_struct, distributed)

        res = cloud_rest_client.v2.update_deployment(
            cluster=deployment_schema.cluster.host_cluster_display_name,
            name=name,
            update_schema=config_struct,
        )
        if res is None:
            raise NotFound(f"deployment {name} is not found")
        logger.debug("Deployment Schema: %s", config_struct)
        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def apply(
        cls,
        name: str | None = None,
        bento: Tag | str | None = None,
        cluster: str | None = None,
        config_dict: dict[str, t.Any] | None = None,
        config_file: t.TextIO | str | None = None,
        path_context: str | None = None,
        context: str | None = None,
    ) -> DeploymentInfo:
        if name is None:
            raise ValueError("name is required")
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.get_deployment(name, cluster)
        if res is None:
            if bento is not None:
                return cls.create(
                    bento=bento,
                    name=name,
                    cluster=cluster,
                    config_dict=config_dict,
                    config_file=config_file,
                    path_context=path_context,
                    context=context,
                )
            else:
                raise ValueError("bento is required")
        if bento is None:
            bento = cls._convert_schema_to_bento(_schema=res)

        schema_dict: dict[str, t.Any] = {"bento": str(bento)}
        distributed = res.distributed

        if isinstance(config_file, str):
            real_path = resolve_user_filepath(config_file, path_context)
            try:
                with open(real_path, "r") as file:
                    config_dict = yaml.safe_load(file)
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
                config_dict = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML config-file stream: %s", exc)
                raise
        if config_dict is None:
            raise BentoMLException("Apply a deployment needs a configuration input")

        schema_dict.update(config_dict)
        config_struct = bentoml_cattr.structure(schema_dict, UpdateDeploymentSchemaV2)
        cls._fix_and_validate_schema(config_struct, distributed)

        res = cloud_rest_client.v2.update_deployment(
            name=name,
            update_schema=config_struct,
            cluster=res.cluster.host_cluster_display_name,
        )
        if res is None:
            raise NotFound(f"deployment {name} is not found")
        logger.debug("Deployment Schema: %s", config_struct)
        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def get(
        cls,
        name: str,
        cluster: str | None = None,
        context: str | None = None,
    ) -> DeploymentInfo:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.get_deployment(name, cluster)
        if res is None:
            raise NotFound(f"deployment {name} is not found")
        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def terminate(
        cls,
        name: str,
        cluster: str | None = None,
        context: str | None = None,
    ) -> DeploymentInfo:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.terminate_deployment(name, cluster)
        if res is None:
            raise NotFound(f"Deployment {name} is not found")
        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def delete(
        cls,
        name: str,
        cluster: str | None = None,
        context: str | None = None,
    ) -> None:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.delete_deployment(name, cluster)
        if res is None:
            raise NotFound(f"Deployment {name} is not found")
