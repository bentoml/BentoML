from __future__ import annotations

import base64
import logging
import time
import typing as t
from os import path
from threading import Event
from threading import Thread

import attr
import rich
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
from ..bento.bento import BentoInfo
from ..configuration.containers import BentoMLContainer
from ..tag import Tag
from ..utils import bentoml_cattr
from ..utils import resolve_user_filepath
from .base import Spinner
from .config import get_rest_api_client
from .schemas.modelschemas import DeploymentStatus
from .schemas.modelschemas import DeploymentTargetHPAConf
from .schemas.schemasv2 import CreateDeploymentSchema as CreateDeploymentSchemaV2
from .schemas.schemasv2 import DeploymentSchema
from .schemas.schemasv2 import DeploymentTargetSchema
from .schemas.schemasv2 import KubePodSchema
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


@attr.define
class DeploymentConfigParameters:
    name: str | None = None
    path_context: str | None = None
    context: str | None = None
    bento: Tag | str | None = None
    cluster: str | None = None
    access_authorization: bool | None = None
    scaling_min: int | None = None
    scaling_max: int | None = None
    instance_type: str | None = None
    strategy: str | None = None
    envs: t.List[dict[str, t.Any]] | None = None
    secrets: t.List[str] | None = (None,)
    extras: dict[str, t.Any] | None = None
    config_dict: dict[str, t.Any] | None = None
    config_file: str | t.TextIO | None = None
    cli: bool = False
    service_name: str | None = None
    cfg_dict: dict[str, t.Any] | None = None
    _param_config: dict[str, t.Any] | None = None

    def verify(
        self,
    ):
        deploy_by_param = (
            self.name
            or self.bento
            or self.cluster
            or self.access_authorization
            or self.scaling_min
            or self.scaling_max
            or self.instance_type
            or self.strategy
            or self.envs
            or self.secrets
            or self.extras
        )

        if (
            self.config_dict
            and self.config_file
            or self.config_dict
            and deploy_by_param
            or self.config_file
            and deploy_by_param
        ):
            raise BentoMLException(
                "Configure a deployment can only use one of the following: config_dict, config_file, or the other parameters"
            )

        if deploy_by_param:
            self.cfg_dict = {
                k: v
                for k, v in [
                    ("name", self.name),
                    ("bento", self.bento),
                    ("cluster", self.cluster),
                    ("access_authorization", self.access_authorization),
                    ("envs", self.envs if self.envs else None),
                    ("secrets", self.secrets),
                ]
                if v is not None
            }
            # add service name
            self._param_config = {
                k: v
                for k, v in [
                    ("scaling_min", self.scaling_min),
                    ("scaling_max", self.scaling_max),
                    ("instance_type", self.instance_type),
                    ("strategy", self.strategy),
                    ("extras", self.extras),
                ]
                if v is not None
            }
        elif self.config_dict:
            self.cfg_dict = self.config_dict
        elif isinstance(self.config_file, str):
            real_path = resolve_user_filepath(self.config_file, self.path_context)
            try:
                with open(real_path, "r") as file:
                    self.cfg_dict = yaml.safe_load(file)
            except FileNotFoundError:
                raise ValueError(f"File not found: {real_path}")
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML file: %s", exc)
                raise
            except Exception as e:
                raise ValueError(
                    f"An error occurred while reading the file: {real_path}\n{e}"
                )
        elif self.config_file:
            try:
                self.cfg_dict = yaml.safe_load(self.config_file)
            except yaml.YAMLError as exc:
                logger.error("Error while parsing YAML config-file stream: %s", exc)
                raise
        else:
            raise BentoMLException(
                "Must provide either config_dict, config_file, or the other parameters"
            )

        if self.cfg_dict is None:
            self.cfg_dict = {}

        bento_name = self.cfg_dict.get("bento")
        # determine if bento is a path or a name
        if bento_name:
            if isinstance(bento_name, str) and path.exists(bento_name):
                # target is a path
                if self.cli:
                    rich.print(f"building bento from [green]{bento_name}[/] ...")
                bento_info = get_bento_info(
                    project_path=bento_name,
                    context=self.context,
                    cli=self.cli,
                )
            else:
                if self.cli:
                    rich.print(f"using bento [green]{bento_name}[/]...")
                bento_info = get_bento_info(
                    bento=str(bento_name),
                    context=self.context,
                    cli=self.cli,
                )
            self.cfg_dict["bento"] = bento_info.tag
            if self.service_name is None:
                self.service_name = bento_info.entry_service

    def get_name(self):
        if self.cfg_dict is None:
            raise BentoMLException(
                "DeploymentConfigParameters.verify() must be called first"
            )
        return self.cfg_dict.get("name")

    def get_cluster(self, pop: bool = True):
        if self.cfg_dict is None:
            raise BentoMLException(
                "DeploymentConfigParameters.verify() must be called first"
            )
        if pop:
            return self.cfg_dict.pop("cluster", None)
        else:
            return self.cfg_dict.get("cluster")

    def get_config_dict(self, bento: str | None = None):
        if self.cfg_dict is None:
            raise BentoMLException(
                "DeploymentConfigParameters.verify() must be called first"
            )
        if self.service_name is None:
            if bento is None:
                if self.cfg_dict.get("bento") is None:
                    raise BentoMLException("Bento is required")
                bento = self.cfg_dict.get("bento")

            info = get_bento_info(
                bento=bento,
                context=self.context,
            )
            if info.entry_service == "":
                # for compatibility
                self.service_name = "apiserver"
            else:
                self.service_name = info.entry_service
        if self._param_config is not None:
            scaling_min = self._param_config.pop("scaling_min", None)
            scaling_max = self._param_config.pop("scaling_max", None)
            if scaling_min is not None or scaling_max is not None:
                self._param_config["scaling"] = {
                    "min_replicas": scaling_min,
                    "max_replicas": scaling_max,
                }
                self._param_config["scaling"] = {
                    k: v
                    for k, v in self._param_config["scaling"].items()
                    if v is not None
                }

            strategy = self._param_config.pop("strategy", None)
            if strategy is not None:
                self._param_config["deployment_strategy"] = strategy
            self.cfg_dict["services"] = {self.service_name: self._param_config}

        return self.cfg_dict


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
def get_bento_info(
    project_path: str | None = None,
    bento: str | Tag | None = None,
    cli: bool = False,
    context: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> BentoInfo:
    if project_path:
        from bentoml.bentos import build_bentofile

        if cli:
            with Spinner() as spinner:
                with spinner.spin(
                    text=f"🍱 Building bento from project: {project_path}"
                ):
                    bento_obj = build_bentofile(
                        build_ctx=project_path, _bento_store=_bento_store
                    )
                    spinner.log(f'🍱 Built bento "{bento_obj.info.tag}"')
        else:
            bento_obj = build_bentofile(
                build_ctx=project_path, _bento_store=_bento_store
            )

        _cloud_client.push_bento(bento=bento_obj, context=context)
        return bento_obj.info
    elif bento:
        bento = Tag.from_taglike(bento)
        try:
            bento_obj = _bento_store.get(bento)
        except NotFound:
            bento_obj = None

        # try to get from bentocloud
        try:
            bento_schema = _cloud_client.get_bento(
                context=context, name=bento.name, version=bento.version
            )
        except NotFound:
            bento_schema = None

        if bento_obj is not None:
            # push to bentocloud
            _cloud_client.push_bento(bento=bento_obj, context=context)
            return bento_obj.info
        if bento_schema is not None:
            assert bento_schema.manifest is not None
            if cli:
                with _cloud_client.spinner as spinner:
                    spinner.log(
                        f"[bold blue]Using bento {bento.name}:{bento.version} from bentocloud to deploy"
                    )
            return BentoInfo(
                tag=Tag(name=bento.name, version=bento.version),
                entry_service=bento_schema.manifest.entry_service,
                service=bento_schema.manifest.service,
            )
        raise NotFound(f"bento {bento} not found in both local and cloud")
    else:
        raise BentoMLException(
            "Create a deployment needs a target; project path or bento is necessary"
        )


@attr.define
class DeploymentConfig(CreateDeploymentSchemaV2):
    def to_yaml(self, with_meta: bool = True):
        config_dict = self.to_dict(with_meta=with_meta)
        return yaml.dump(config_dict, sort_keys=False)

    def to_dict(self, with_meta: bool = True):
        config_dict = bentoml_cattr.unstructure(self)
        if with_meta is False:
            # delete name and bento
            config_dict.pop("name", None)
            config_dict.pop("bento", None)

        return config_dict


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
    _context: t.Optional[str] = attr.field(alias="_context", repr=False)
    _schema: DeploymentSchema = attr.field(alias="_schema", repr=False)
    _urls: t.Optional[list[str]] = attr.field(alias="_urls", default=None, repr=False)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "bento": self.get_bento(refetch=False),
            "cluster": self.cluster,
            "endpoint_urls": self._urls if self._urls else None,
            "admin_console": self.admin_console,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "config": (
                config.to_dict(with_meta=False)
                if (config := self.get_config(refetch=False)) is not None
                else None
            ),
            "status": self.get_status(refetch=False).to_dict(),
        }

    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

    def _refetch(self) -> None:
        res = Deployment.get(self.name, self.cluster, self._context)
        self._schema = res._schema
        self._urls = res._urls

    def _refetch_target(self, refetch: bool) -> DeploymentTargetSchema | None:
        if refetch:
            self._refetch()
        if self._schema.latest_revision is None:
            return None
        if len(self._schema.latest_revision.targets) == 0:
            return None
        return self._schema.latest_revision.targets[0]

    def get_config(self, refetch: bool = True) -> DeploymentConfig | None:
        target = self._refetch_target(refetch)
        if target is None:
            return None
        if target.config is None:
            return None

        return DeploymentConfig(
            name=self.name,
            bento=self.get_bento(refetch=False),
            services=target.config.services,
            access_authorization=target.config.access_authorization,
            envs=target.config.envs,
        )

    def get_status(self, refetch: bool = True) -> DeploymentState:
        if refetch:
            self._refetch()
        updated_at = self._schema.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if self._schema.updated_at is not None:
            updated_at = self._schema.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        return DeploymentState(
            status=self._schema.status,
            created_at=self._schema.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=updated_at,
        )

    def get_endpoint_urls(self, refetch: bool = True) -> list[str]:
        if refetch:
            self._refetch()
        if self._urls is None or len(self._urls) != 1:
            raise BentoMLException("Deployment endpoint url is not ready")
        return self._urls

    def get_bento(self, refetch: bool = True) -> str:
        target = self._refetch_target(refetch)
        if target is None:
            return ""
        if target.bento is None:
            return ""
        return target.bento.repository.name + ":" + target.bento.version

    def get_client(
        self,
        media_type: str = "application/json",
        token: str | None = None,
    ) -> SyncHTTPClient:
        from _bentoml_impl.client import SyncHTTPClient

        self._refetch()
        if (
            self._schema.status != DeploymentStatus.Running.value
            and self._schema.status != DeploymentStatus.ScaledToZero.value
        ):
            raise BentoMLException(f"Deployment status is {self._schema.status}")
        if self._urls is None or len(self._urls) != 1:
            raise BentoMLException("Deployment url is not ready")

        return SyncHTTPClient(self._urls[0], token=token)

    def get_async_client(
        self,
        media_type: str = "application/json",
        token: str | None = None,
    ) -> AsyncHTTPClient:
        from _bentoml_impl.client import AsyncHTTPClient

        self._refetch()
        if (
            self._schema.status != DeploymentStatus.Running.value
            and self._schema.status != DeploymentStatus.ScaledToZero.value
        ):
            raise BentoMLException(f"Deployment status is {self._schema.status}")
        if self._urls is None or len(self._urls) != 1:
            raise BentoMLException("Deployment url is not ready")
        return AsyncHTTPClient(self._urls[0], token=token)

    def wait_until_ready(
        self,
        timeout: int = 3600,
        check_interval: int = 10,
        spinner: Spinner | None = None,
    ) -> None:
        from httpx import TimeoutException

        start_time = time.time()
        if spinner is not None:
            stop_tail_event = Event()

            def tail_image_builder_logs() -> None:
                cloud_rest_client = get_rest_api_client(self._context)
                pod: KubePodSchema | None = None
                while True:
                    pod = cloud_rest_client.v2.get_deployment_image_builder_pod(
                        self.name, self.cluster
                    )
                    if pod is None:
                        if stop_tail_event.wait(check_interval):
                            return
                        continue
                    if pod.pod_status.status == "Running":
                        break
                    if stop_tail_event.wait(check_interval):
                        return

                is_first = True
                logs_tailer = cloud_rest_client.v2.tail_logs(
                    cluster_name=self.cluster,
                    namespace=self._schema.kube_namespace,
                    pod_name=pod.name,
                    container_name="builder",
                    stop_event=stop_tail_event,
                )

                for piece in logs_tailer:
                    decoded_bytes = base64.b64decode(piece)
                    decoded_str = decoded_bytes.decode("utf-8")
                    if is_first:
                        is_first = False
                        spinner.update("🚧 Image building...")
                        spinner.stop()
                    print(decoded_str, end="", flush=True)

            tail_thread: Thread | None = None

            try:
                status: DeploymentState | None = None
                spinner.update(
                    f'🔄 Waiting for deployment "{self.name}" to be ready...'
                )
                while time.time() - start_time < timeout:
                    for _ in range(3):
                        try:
                            status = self.get_status()
                            break
                        except TimeoutException:
                            spinner.update(
                                "⚠️ Unable to get deployment status, retrying..."
                            )
                    if status is None:
                        spinner.log(
                            "🚨 [bold red]Unable to contact the server, but the deployment is created. You can check the status on the bentocloud website.[/bold red]"
                        )
                        return
                    spinner.update(
                        f'🔄 Waiting for deployment "{self.name}" to be ready. Current status: "{status.status}"'
                    )
                    if status.status == DeploymentStatus.ImageBuilding.value:
                        if tail_thread is None:
                            tail_thread = Thread(
                                target=tail_image_builder_logs, daemon=True
                            )
                            tail_thread.start()

                    if status.status in (
                        DeploymentStatus.Running.value,
                        DeploymentStatus.ScaledToZero.value,
                    ):
                        spinner.stop()
                        spinner.console.print(
                            f'✅ [bold green] Deployment "{self.name}" is ready:[/] {self.admin_console}'
                        )
                        return
                    if status.status in [
                        DeploymentStatus.Failed.value,
                        DeploymentStatus.ImageBuildFailed.value,
                        DeploymentStatus.Terminated.value,
                        DeploymentStatus.Terminating.value,
                        DeploymentStatus.Unhealthy.value,
                    ]:
                        spinner.stop()
                        spinner.console.print(
                            f'🚨 [bold red]Deployment "{self.name}" is not ready. Current status: "{status.status}"[/]'
                        )
                        return

                    time.sleep(check_interval)

                spinner.stop()
                spinner.console.print(
                    f'🚨 [bold red]Time out waiting for Deployment "{self.name}" ready[/]'
                )
                return
            finally:
                stop_tail_event.set()
                if tail_thread is not None:
                    tail_thread.join()
        else:
            while time.time() - start_time < timeout:
                status: DeploymentState | None = None
                for _ in range(3):
                    try:
                        status = self.get_status()
                        break
                    except TimeoutException:
                        pass
                if status is None:
                    logger.error(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unable to contact the server, but the deployment is created. You can check the status on the bentocloud website."
                    )
                    return
                if status.status in (
                    DeploymentStatus.Running.value,
                    DeploymentStatus.ScaledToZero.value,
                ):
                    logger.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Deployment '{self.name}' is ready."
                    )
                    return
                logger.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for deployment '{self.name}' to be ready. Current status: '{status.status}'."
                )
                time.sleep(check_interval)

        logger.error(f"Timed out waiting for deployment '{self.name}' to be ready.")


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
            raise BentoMLException(
                "min scaling values must be less than or equal to max scaling values"
            )
        if scaling.min_replicas < 0:
            raise BentoMLException(
                "min scaling values must be greater than or equal to 0"
            )
        if scaling.max_replicas <= 0:
            raise BentoMLException("max scaling values must be greater than 0")
        return scaling

    @classmethod
    def _fix_and_validate_schema(
        cls,
        config_struct: UpdateDeploymentSchemaV2,
    ):
        # fix scaling
        for _, svc in config_struct.services.items():
            svc.scaling = cls._fix_scaling(svc.scaling)

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
            access_authorization=target_schema.config.access_authorization,
            envs=target_schema.config.envs,
            bento=target_schema.bento.repository.name + ":" + target_schema.bento.name,
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
        return DeploymentInfo(
            name=res.name,
            admin_console=f"{client.v1.endpoint}/deployments/{res.name}?cluster={res.cluster.name}&namespace={res.kube_namespace}",
            created_at=res.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            created_by=res.creator.name,
            cluster=res.cluster.name,
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
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v2.list_deployment(
                search=search, count=res_count.total, all=True
            )
        else:
            res_count = cloud_rest_client.v2.list_deployment(cluster, search=search)
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v2.list_deployment(
                cluster, search=search, count=res_count.total
            )
        return [cls._generate_deployment_info_(context, schema) for schema in res.items]

    @classmethod
    def create(
        cls,
        deployment_config_params: DeploymentConfigParameters,
        context: str | None = None,
    ) -> DeploymentInfo:
        cluster = deployment_config_params.get_cluster()
        if (
            deployment_config_params.cfg_dict is None
            or deployment_config_params.cfg_dict.get("bento") is None
        ):
            raise ValueError("bento is required")

        config_params = deployment_config_params.get_config_dict()
        config_struct = bentoml_cattr.structure(config_params, CreateDeploymentSchemaV2)
        cls._fix_and_validate_schema(config_struct)

        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.create_deployment(
            create_schema=config_struct, cluster=cluster
        )

        logger.debug("Deployment Schema: %s", config_struct)

        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def update(
        cls,
        deployment_config_params: DeploymentConfigParameters,
        context: str | None = None,
    ) -> DeploymentInfo:
        name = deployment_config_params.get_name()
        if name is None:
            raise ValueError("name is required")
        cluster = deployment_config_params.get_cluster()

        cloud_rest_client = get_rest_api_client(context)
        deployment_schema = cloud_rest_client.v2.get_deployment(name, cluster)
        orig_dict = cls._convert_schema_to_update_schema(deployment_schema)

        config_params = deployment_config_params.get_config_dict(
            orig_dict.get("bento"),
        )

        config_merger.merge(orig_dict, config_params)
        config_struct = bentoml_cattr.structure(orig_dict, UpdateDeploymentSchemaV2)

        cls._fix_and_validate_schema(config_struct)

        res = cloud_rest_client.v2.update_deployment(
            cluster=deployment_schema.cluster.name,
            name=name,
            update_schema=config_struct,
        )
        logger.debug("Deployment Schema: %s", config_struct)
        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def apply(
        cls,
        deployment_config_params: DeploymentConfigParameters,
        context: str | None = None,
    ) -> DeploymentInfo:
        cloud_rest_client = get_rest_api_client(context)
        name = deployment_config_params.get_name()
        if name is not None:
            try:
                deployment_schema = cloud_rest_client.v2.get_deployment(
                    name, deployment_config_params.get_cluster(pop=False)
                )
            except NotFound:
                return cls.create(
                    deployment_config_params=deployment_config_params,
                    context=context,
                )
            # directly update the deployment with the schema, do not merge with the existing schema
            if deployment_config_params.get_name() != deployment_schema.name:
                raise BentoMLException(
                    f"Deployment name cannot be changed, current name is {deployment_schema.name}"
                )
            if (
                deployment_config_params.get_cluster(pop=False)
                != deployment_schema.cluster.name
            ):
                raise BentoMLException(
                    f"Deployment cluster cannot be changed, current cluster is {deployment_schema.cluster.name}"
                )
            config_struct = bentoml_cattr.structure(
                deployment_config_params.get_config_dict(), UpdateDeploymentSchemaV2
            )
            cls._fix_and_validate_schema(config_struct)

            res = cloud_rest_client.v2.update_deployment(
                name=name,
                update_schema=config_struct,
                cluster=deployment_schema.cluster.name,
            )
            logger.debug("Deployment Schema: %s", config_struct)
            return cls._generate_deployment_info_(context, res, res.urls)

        return cls.create(
            deployment_config_params=deployment_config_params,
            context=context,
        )

    @classmethod
    def get(
        cls,
        name: str,
        cluster: str | None = None,
        context: str | None = None,
    ) -> DeploymentInfo:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.get_deployment(name, cluster)
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
        return cls._generate_deployment_info_(context, res, res.urls)

    @classmethod
    def delete(
        cls,
        name: str,
        cluster: str | None = None,
        context: str | None = None,
    ) -> None:
        cloud_rest_client = get_rest_api_client(context)
        cloud_rest_client.v2.delete_deployment(name, cluster)

    @classmethod
    def list_instance_types(
        cls,
        cluster: str | None = None,
        context: str | None = None,
    ) -> list[InstanceTypeInfo]:
        cloud_rest_client = get_rest_api_client(context)
        res = cloud_rest_client.v2.list_instance_types(cluster)
        return [
            InstanceTypeInfo(
                name=schema.display_name,
                description=schema.description,
                cpu=(
                    schema.config.resources.requests.cpu
                    if schema.config.resources and schema.config.resources.requests
                    else None
                ),
                memory=(
                    schema.config.resources.requests.memory
                    if schema.config.resources and schema.config.resources.requests
                    else None
                ),
                gpu=(
                    schema.config.resources.requests.gpu
                    if schema.config.resources and schema.config.resources.requests
                    else None
                ),
                gpu_type=(
                    schema.config.gpu_config.type if schema.config.gpu_config else None
                ),
                price=schema.config.price,
            )
            for schema in res
        ]


@attr.define
class InstanceTypeInfo:
    name: t.Optional[str] = None
    price: t.Optional[str] = None
    description: t.Optional[str] = None
    cpu: t.Optional[str] = None
    memory: t.Optional[str] = None
    gpu: t.Optional[str] = None
    gpu_type: t.Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in attr.asdict(self).items() if v is not None and v != ""}
