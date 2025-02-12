from __future__ import annotations

import base64
import contextlib
import hashlib
import logging
import os
import time
import typing as t
from pathlib import Path
from threading import Event
from threading import Thread

import attr
import fs
import rich
import yaml
from deepmerge.merger import Merger
from pathspec import PathSpec
from rich.console import Console
from simple_di import Provide
from simple_di import inject

from ..bento.bento import DEFAULT_BENTO_BUILD_FILES
from ..bento.bento import Bento
from ..bento.build_config import BentoBuildConfig
from ..configuration import is_editable_bentoml
from ..utils.pkg import source_locations

if t.TYPE_CHECKING:
    from _bentoml_impl.client import AsyncHTTPClient
    from _bentoml_impl.client import SyncHTTPClient

    from ..bento.bento import BentoStore
    from .client import RestApiClient

from ...exceptions import BentoMLException
from ...exceptions import NotFound
from ..configuration.containers import BentoMLContainer
from ..tag import Tag
from ..utils import filter_control_codes
from ..utils import is_jupyter
from ..utils.cattr import bentoml_cattr
from ..utils.filesystem import resolve_user_filepath
from .base import Spinner
from .schemas.modelschemas import BentoManifestSchema
from .schemas.modelschemas import DeploymentStatus
from .schemas.modelschemas import DeploymentTargetHPAConf
from .schemas.schemasv2 import CreateDeploymentSchema as CreateDeploymentSchemaV2
from .schemas.schemasv2 import DeleteDeploymentFilesSchema
from .schemas.schemasv2 import DeploymentFileListSchema
from .schemas.schemasv2 import DeploymentSchema
from .schemas.schemasv2 import DeploymentTargetSchema
from .schemas.schemasv2 import KubePodSchema
from .schemas.schemasv2 import UpdateDeploymentSchema as UpdateDeploymentSchemaV2
from .schemas.schemasv2 import UploadDeploymentFilesSchema

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
    bento: Tag | str | None = None
    cluster: str | None = None
    access_authorization: bool | None = None
    scaling_min: int | None = None
    scaling_max: int | None = None
    instance_type: str | None = None
    strategy: str | None = None
    envs: t.List[dict[str, t.Any]] | None = None
    secrets: t.List[str] | None = None
    extras: dict[str, t.Any] | None = None
    config_dict: dict[str, t.Any] | None = None
    config_file: str | t.TextIO | None = None
    cli: bool = False
    dev: bool = False
    service_name: str | None = None
    cfg_dict: dict[str, t.Any] | None = None
    _param_config: dict[str, t.Any] | None = None

    def verify(self, create: bool = True):
        from bentoml._internal.configuration.containers import BentoMLContainer

        from .secret import SecretAPI

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
                    ("dev", self.dev),
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
            if isinstance(bento_name, str) and os.path.exists(bento_name):
                # target is a path
                if self.cli:
                    rich.print(f"building bento from [green]{bento_name}[/] ...")
                bento_info = ensure_bento(
                    project_path=bento_name, bare=self.dev, cli=self.cli
                )
            elif self.dev:  # dev mode and bento is built
                return
            else:
                if self.cli:
                    rich.print(f"using bento [green]{bento_name}[/]...")
                bento_info = ensure_bento(bento=str(bento_name), cli=self.cli)
            if create:
                manifest = (
                    bento_info.get_manifest()
                    if isinstance(bento_info, Bento)
                    else bento_info
                )
                required_envs = [env.name for env in manifest.envs if not env.value]
                provided_envs: list[str] = [env["name"] for env in (self.envs or [])]
                if self.secrets:
                    secret_api = SecretAPI(BentoMLContainer.rest_api_client.get())
                    for secret_name in self.secrets:
                        secret = secret_api.get(secret_name, cluster=self.cluster)
                        if secret.content.type == "env":
                            provided_envs.extend(
                                item.key for item in secret.content.items
                            )

                missing_envs = [
                    env for env in required_envs if env not in provided_envs
                ]
                if missing_envs:
                    raise BentoMLException(
                        f"Environment variables must be provided for bento but missing: {missing_envs}"
                    )
            self.cfg_dict["bento"] = str(bento_info.tag)
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

            info = ensure_bento(bento=bento)
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
def ensure_bento(
    project_path: str | None = None,
    bento: str | Tag | None = None,
    cli: bool = False,
    bare: bool = False,
    push: bool = True,
    reload: bool = False,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _client: RestApiClient = Provide[BentoMLContainer.rest_api_client],
) -> Bento | BentoManifestSchema:
    from bentoml.bentos import build_bentofile

    from .bento import BentoAPI

    if not project_path and not bento:
        raise BentoMLException(
            "Creating a deployment needs a target; project path or bento is necessary"
        )
    bento_api = BentoAPI(_client)
    if project_path:
        bento_obj = build_bentofile(
            build_ctx=project_path, bare=bare, _bento_store=_bento_store, reload=reload
        )
        if cli:
            rich.print(f":bento: Built bento [green]{bento_obj.info.tag}[/]")
        if push:
            bento_api.push(bento=bento_obj, bare=bare)
        return bento_obj
    assert bento is not None
    bento_tag = Tag.from_taglike(bento)
    try:
        bento_obj = _bento_store.get(bento_tag)
    except NotFound:
        bento_obj = None

    # try to get from bentocloud
    try:
        bento_schema = bento_api.get(name=bento_tag.name, version=bento_tag.version)
    except NotFound:
        bento_schema = None

    if bento_obj is not None:
        # push to bentocloud
        if push:
            bento_api.push(bento=bento_obj, bare=bare)
        return bento_obj
    if bento_schema is not None:
        assert bento_schema.manifest is not None
        if cli:
            rich.print(
                f"[bold blue]Using bento [green]{bento_tag}[/] from bentocloud to deploy"
            )
        bento_schema.manifest.version = bento_tag.version
        return bento_schema.manifest

    # bento is a service definition
    if isinstance(bento, str):
        try:
            bento_obj = build_bentofile(
                service=bento, bare=bare, _bento_store=_bento_store, reload=reload
            )
        except BentoMLException:
            pass
        else:
            if cli:
                rich.print(f":bento: Built bento [green]{bento_obj.info.tag}[/]")
            if push:
                bento_api.push(bento=bento_obj, bare=bare)
            return bento_obj
    raise NotFound(f"Bento {bento} is not found in both local and bentocloud")


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
class Deployment:
    name: str
    admin_console: str
    created_at: str
    created_by: str
    cluster: str
    _client: RestApiClient = attr.field(alias="_client", repr=False)
    _schema: DeploymentSchema = attr.field(alias="_schema", repr=False)
    _urls: t.Optional[list[str]] = attr.field(alias="_urls", default=None, repr=False)

    @property
    def is_dev(self) -> bool:
        return self._schema.manifest is not None and self._schema.manifest.dev

    def to_dict(self) -> dict[str, t.Any]:
        result = {
            "name": self.name,
            "bento": self.get_bento(refetch=False),
            "cluster": self.cluster,
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

        # Add "endpoint_urls" only if it's not None
        if self._urls:
            result["endpoint_urls"] = self._urls

        return result

    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

    def _refetch(self) -> None:
        res = DeploymentAPI(self._client).get(self.name, self.cluster)
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
        spinner: Spinner | None = None,
        timeout: int = 3600,
        check_interval: int = 10,
    ) -> int:
        from httpx import TimeoutException

        start_time = time.time()
        stop_tail_event = Event()
        if spinner is None:
            spinner = Spinner()
        console = spinner.console

        def tail_image_builder_logs() -> None:
            started_at = time.time()
            wait_pod_timeout = 60 * 10
            pod: KubePodSchema | None = None
            while True:
                pod = self._client.v2.get_deployment_image_builder_pod(
                    self.name, self.cluster
                )
                if pod is None:
                    if time.time() - started_at > timeout:
                        spinner.console.print(
                            ":police_car_light: [bold red]Time out waiting for image builder pod created[/bold red]"
                        )
                        return
                    if stop_tail_event.wait(check_interval):
                        return
                    continue
                if pod.pod_status.status == "Running":
                    break
                if time.time() - started_at > wait_pod_timeout:
                    spinner.console.print(
                        ":police_car_light: [bold red]Time out waiting for image builder pod running[/bold red]"
                    )
                    return
                if stop_tail_event.wait(check_interval):
                    return

            is_first = True
            logs_tailer = self._client.v2.tail_logs(
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
                    spinner.update(":construction: Image building...")
                    spinner.stop()
                console.print(decoded_str, end="", markup=False, highlight=False)

        tail_thread: Thread | None = None

        try:
            status: DeploymentState | None = None
            spinner.update(
                f':hourglass: Waiting for deployment "{self.name}" to be ready...'
            )
            while time.time() - start_time < timeout:
                for _ in range(3):
                    try:
                        new_status = self.get_status()
                        break
                    except TimeoutException:
                        spinner.update(
                            ":warning: Unable to get deployment status, retrying..."
                        )
                else:
                    spinner.log(
                        ":police_car_light: [bold red]Unable to contact the server, but the deployment is created. "
                        "You can check the status on the bentocloud website.[/bold red]"
                    )
                    return 1
                if (
                    status is None or status.status != new_status.status
                ):  # on status change
                    status = new_status
                    spinner.update(
                        f':hourglass: Waiting for deployment "{self.name}" to be ready. Current status: "{status.status}"'
                    )
                    if status.status == DeploymentStatus.ImageBuilding.value:
                        if tail_thread is None and not is_jupyter():
                            tail_thread = Thread(
                                target=tail_image_builder_logs, daemon=True
                            )
                            tail_thread.start()
                    elif (
                        tail_thread is not None
                    ):  # The status has changed from ImageBuilding to other
                        stop_tail_event.set()
                        tail_thread.join()
                        tail_thread = None
                        spinner.start()

                if status.status in [
                    DeploymentStatus.Failed.value,
                    DeploymentStatus.ImageBuildFailed.value,
                    DeploymentStatus.Terminated.value,
                    DeploymentStatus.Terminating.value,
                    DeploymentStatus.Unhealthy.value,
                ]:
                    spinner.stop()
                    console.print(
                        f':police_car_light: [bold red]Deployment "{self.name}" is not ready. Current status: "{status.status}"[/]'
                    )
                    return 1
                if status.status in (
                    DeploymentStatus.Running.value,
                    DeploymentStatus.ScaledToZero.value,
                ):
                    spinner.stop()
                    console.print(
                        f':white_check_mark: [bold green] Deployment "{self.name}" is ready:[/] {self.admin_console}'
                    )
                    break

                time.sleep(check_interval)
            else:
                spinner.stop()
                console.print(
                    f':police_car_light: [bold red]Time out waiting for Deployment "{self.name}" ready[/]'
                )
                return 1

        finally:
            stop_tail_event.set()
            if tail_thread is not None:
                tail_thread.join()
        return 0

    def upload_files(
        self, files: t.Iterable[tuple[str, bytes]], *, console: Console | None = None
    ) -> None:
        console = console or rich.get_console()
        all_files = [
            {
                "path": path,
                "b64_encoded_content": base64.b64encode(content).decode("utf-8"),
            }
            for path, content in files
        ]
        max_chunk_size = 64 * 1024 * 1024  # 64 Mb
        current_size = 0
        chunk: list[dict[str, t.Any]] = []
        for file in all_files:
            console.print(f" [green]Uploading[/] {file['path']}")
            chunk.append(file)
            current_size += len(file["b64_encoded_content"])
            if current_size >= max_chunk_size:
                self._client.v2.upload_files(
                    self.name,
                    bentoml_cattr.structure(
                        {"files": chunk}, UploadDeploymentFilesSchema
                    ),
                    cluster=self.cluster,
                )
                chunk.clear()
                current_size = 0
        if chunk:
            self._client.v2.upload_files(
                self.name,
                bentoml_cattr.structure({"files": chunk}, UploadDeploymentFilesSchema),
                cluster=self.cluster,
            )

    def delete_files(
        self, paths: t.Iterable[str], *, console: Console | None = None
    ) -> None:
        console = console or rich.get_console()
        paths = list(paths)
        for path in paths:
            console.print(f" [red]Deleting[/] {path}")
        data = {"paths": paths}
        self._client.v2.delete_files(
            self.name,
            bentoml_cattr.structure(data, DeleteDeploymentFilesSchema),
            cluster=self.cluster,
        )

    def list_files(self) -> DeploymentFileListSchema:
        return self._client.v2.list_files(self.name, cluster=self.cluster)

    def _init_deployment_files(
        self, bento_dir: str, spinner: Spinner, timeout: int = 600
    ) -> tuple[str, str]:
        from ..bento.build_config import BentoPathSpec

        check_interval = 5
        start_time = time.time()
        console = spinner.console
        spinner_text = ":hourglass: Preparing codespace - status: [green]{}[/]"
        status = self.get_status(False).status
        while time.time() - start_time < timeout:
            spinner.update(spinner_text.format(status))
            if status in [
                DeploymentStatus.Failed.value,
                DeploymentStatus.Terminated.value,
                DeploymentStatus.Terminating.value,
                DeploymentStatus.Unhealthy.value,
            ]:
                raise BentoMLException(
                    f"Deployment {self.name} aborted. Current status: {status}"
                )

            pods = self._client.v2.list_deployment_pods(self.name, self.cluster)
            if any(
                pod.labels.get("yatai.ai/bento-function-component-type") == "api-server"
                and pod.status.phase == "Running"
                and pod.pod_status.status != "Terminating"
                for pod in pods
            ):
                break
            time.sleep(check_interval)
            status = self.get_status(True).status
        else:
            raise TimeoutError("Timeout waiting for API server pod to be ready")

        build_config = BentoBuildConfig.from_bento_dir(bento_dir)
        bento_spec = BentoPathSpec(
            build_config.include, build_config.exclude, bento_dir
        )
        upload_files: list[tuple[str, bytes]] = []
        requirements_content = _build_requirements_txt(bento_dir)

        pod_files = {file.path: file.md5 for file in self.list_files().files}
        for root, _, files in os.walk(bento_dir):
            for fn in files:
                full_path = os.path.join(root, fn)
                rel_path = os.path.relpath(full_path, bento_dir).replace(os.sep, "/")
                if (
                    not bento_spec.includes(rel_path)
                    and rel_path not in DEFAULT_BENTO_BUILD_FILES
                ):
                    continue
                if rel_path in (REQUIREMENTS_TXT, "setup.sh"):
                    continue
                file_content = open(full_path, "rb").read()
                if (
                    rel_path in pod_files
                    and pod_files[rel_path] == hashlib.md5(file_content).hexdigest()
                ):
                    continue
                upload_files.append((rel_path, file_content))
        if is_editable_bentoml():
            console.print(
                "[yellow]BentoML is installed in editable mode, uploading source code...[/]"
            )
            bentoml_project = Path(source_locations("bentoml")).parent.parent
            for path in EDITABLE_BENTOML_PATHSPEC.match_tree_files(bentoml_project):
                rel_path = os.path.join(EDITABLE_BENTOML_DIR, path)
                file_content = Path(bentoml_project, path).read_bytes()
                if (
                    rel_path in pod_files
                    and pod_files[rel_path] == hashlib.md5(file_content).hexdigest()
                ):
                    continue
                upload_files.append((rel_path, file_content))

        requirements_md5 = hashlib.md5(requirements_content).hexdigest()
        if requirements_md5 != pod_files.get(REQUIREMENTS_TXT, ""):
            upload_files.append((REQUIREMENTS_TXT, requirements_content))
        setup_script = _build_setup_script(bento_dir)
        setup_md5 = hashlib.md5(setup_script).hexdigest()
        if setup_md5 != pod_files.get("setup.sh", ""):
            upload_files.append(("setup.sh", setup_script))
        self.upload_files(upload_files, console=console)
        # Upload a ready flag file after all files are uploaded
        self.upload_files([(".project_ready", b"")], console=console)
        return requirements_md5, setup_md5

    def watch(self, bento_dir: str) -> None:
        import watchfiles

        from ..bento.build_config import BentoPathSpec
        from .bento import BentoAPI

        bento_dir = os.path.abspath(bento_dir)
        build_config = BentoBuildConfig.from_bento_dir(bento_dir)
        deployment_api = DeploymentAPI(self._client)
        bento_api = BentoAPI(self._client)
        bento_spec = BentoPathSpec(
            build_config.include, build_config.exclude, bento_dir
        )
        requirements_hash: str | None = None
        setup_md5: str | None = None
        default_filter = watchfiles.filters.DefaultFilter()
        is_editable = is_editable_bentoml()
        bentoml_project = str(Path(source_locations("bentoml")).parent.parent)
        watch_dirs = [bento_dir]
        if is_editable:
            watch_dirs.append(bentoml_project)

        def watch_filter(change: watchfiles.Change, path: str) -> bool:
            if not default_filter(change, path):
                return False
            if is_editable and fs.path.isparent(bentoml_project, path):
                rel_path = os.path.relpath(path, bentoml_project)
                return EDITABLE_BENTOML_PATHSPEC.match_file(rel_path)
            rel_path = os.path.relpath(path, bento_dir)
            return rel_path in (
                *DEFAULT_BENTO_BUILD_FILES,
                REQUIREMENTS_TXT,
                "setup.sh",
            ) or bento_spec.includes(rel_path)

        console = Console(highlight=False)
        bento_info = ensure_bento(
            bento_dir, bare=True, push=False, reload=True, _client=self._client
        )
        assert isinstance(bento_info, Bento)
        target = self._refetch_target(False)

        def is_bento_changed(bento_info: Bento) -> bool:
            if target is None or target.bento is None:
                return True
            bento_manifest = bento_info.get_manifest()
            if target.bento.manifest == bento_manifest:
                return False
            try:
                from deepdiff import DeepDiff
            except ModuleNotFoundError:
                pass
            else:
                diff = DeepDiff(
                    target.bento.manifest, bento_manifest, ignore_order=True
                )
                console.print(f"[blue]{diff.pretty()}[/]")
            return True

        spinner = Spinner(console=console)
        needs_update = is_bento_changed(bento_info)
        spinner.log(f":laptop_computer: View Dashboard: {self.admin_console}")
        endpoint_url: str | None = None
        stop_event = Event()
        try:
            spinner.start()
            upload_id = spinner.transmission_progress.add_task(
                "Dummy upload task", visible=False
            )
            while True:
                stop_event.clear()
                if needs_update:
                    console.print(":sparkles: [green bold]Bento change detected[/]")
                    spinner.update(":hourglass: Pushing Bento to BentoCloud")
                    bento_api._do_push_bento(bento_info, upload_id, bare=True)  # type: ignore
                    spinner.update(
                        ":hourglass: Updating codespace with new configuration"
                    )
                    update_config = DeploymentConfigParameters(
                        bento=str(bento_info.tag),
                        name=self.name,
                        cluster=self.cluster,
                        cli=False,
                        dev=True,
                    )
                    update_config.verify(create=False)
                    self = deployment_api.update(update_config)
                    target = self._refetch_target(False)
                    needs_update = False
                requirements_hash, setup_md5 = self._init_deployment_files(
                    bento_dir, spinner=spinner
                )
                if endpoint_url is None:
                    endpoint_url = self.get_endpoint_urls(True)[0]
                    spinner.log(f"🌐 Endpoint: {endpoint_url}")
                with self._tail_logs(spinner, stop_event):
                    for changes in watchfiles.watch(
                        *watch_dirs, watch_filter=watch_filter, stop_event=stop_event
                    ):
                        if not is_editable or any(
                            fs.path.isparent(bento_dir, p) for _, p in changes
                        ):
                            try:
                                bento_info = ensure_bento(
                                    bento_dir,
                                    bare=True,
                                    push=False,
                                    reload=True,
                                    _client=self._client,
                                )
                            except Exception as e:
                                spinner.console.print(
                                    f":police_car_light: [bold red]Failed to build Bento: {e}[/]"
                                )
                            else:
                                assert isinstance(bento_info, Bento)
                                if is_bento_changed(bento_info):
                                    # stop log tail and reset the deployment
                                    needs_update = True
                                    break

                        build_config = BentoBuildConfig.from_bento_dir(bento_dir)
                        upload_files: list[tuple[str, bytes]] = []
                        delete_files: list[str] = []
                        affected_files: set[str] = set()

                        for _, path in changes:
                            if is_editable and fs.path.isparent(bentoml_project, path):
                                rel_path = os.path.join(
                                    EDITABLE_BENTOML_DIR,
                                    os.path.relpath(path, bentoml_project),
                                )
                            else:
                                rel_path = os.path.relpath(path, bento_dir)
                                if rel_path == REQUIREMENTS_TXT:
                                    continue
                            if rel_path in affected_files:
                                continue
                            affected_files.add(rel_path)
                            if os.path.exists(path):
                                upload_files.append((rel_path, open(path, "rb").read()))
                            else:
                                delete_files.append(rel_path)
                        setup_script = _build_setup_script(bento_dir)
                        if (
                            new_hash := hashlib.md5(setup_script).hexdigest()
                            != setup_md5
                        ):
                            setup_md5 = new_hash
                            bento_info.tag.version = None
                            bento_info._tag = bento_info.tag.make_new_version()
                            needs_update = True
                            break
                        requirements_content = _build_requirements_txt(bento_dir)
                        if (
                            new_hash := hashlib.md5(requirements_content).hexdigest()
                        ) != requirements_hash:
                            requirements_hash = new_hash
                            upload_files.append(
                                (REQUIREMENTS_TXT, requirements_content)
                            )
                        if upload_files:
                            self.upload_files(upload_files, console=console)
                        if delete_files:
                            self.delete_files(delete_files, console=console)
                        if (status := self.get_status().status) in [
                            DeploymentStatus.Failed.value,
                            DeploymentStatus.Terminated.value,
                            DeploymentStatus.Terminating.value,
                            DeploymentStatus.Unhealthy.value,
                        ]:
                            console.print(
                                f':police_car_light: [bold red]Codespace "{self.name}" aborted. Current status: "{status}"[/]'
                            )
                            return
        except KeyboardInterrupt:
            spinner.log(
                "\nWatcher stopped. Next steps:\n"
                "* Attach to this codespace again:\n"
                f"    [blue]$ bentoml code --attach {self.name} --cluster {self.cluster}[/]\n\n"
                "* Push the bento to BentoCloud:\n"
                "    [blue]$ bentoml build --push[/]\n\n"
                "* Shut down the codespace:\n"
                f"    [blue]$ bentoml deployment terminate {self.name} --cluster {self.cluster}[/]"
            )
        finally:
            spinner.stop()

    @contextlib.contextmanager
    def _tail_logs(
        self, spinner: Spinner, stop_event: Event
    ) -> t.Generator[None, None, None]:
        import itertools
        from collections import defaultdict

        spinner.update("🟡 👀 Watching for changes")
        server_ready = False
        console = spinner.console

        def set_server_ready(is_ready: bool) -> None:
            nonlocal server_ready
            if is_ready is server_ready:
                return
            spinner.update(
                "🟢 👀 Watching for changes"
                if is_ready
                else "🟡 👀 Watching for changes"
            )
            server_ready = is_ready

        pods = self._client.v2.list_deployment_pods(self.name, self.cluster)
        workers: list[Thread] = []

        colors = itertools.cycle(["cyan", "yellow", "blue", "magenta", "green"])
        runner_color: dict[str, str] = defaultdict(lambda: next(colors))

        def heartbeat(event: Event, check_interval: float = 5.0) -> None:
            from httpx import NetworkError
            from httpx import TimeoutException

            endpoint_url = self.get_endpoint_urls(False)[0]
            while not event.is_set():
                try:
                    resp = self._client.session.get(f"{endpoint_url}/readyz", timeout=5)
                except (TimeoutException, NetworkError):
                    set_server_ready(False)
                else:
                    if resp.is_success:
                        set_server_ready(True)
                    else:
                        set_server_ready(False)
                event.wait(check_interval)

        def pod_log_worker(pod: KubePodSchema, stop_event: Event) -> None:
            current = ""
            color = runner_color[pod.runner_name]
            for chunk in self._client.v2.tail_logs(
                cluster_name=self.cluster,
                namespace=self._schema.kube_namespace,
                pod_name=pod.name,
                container_name="main",
                stop_event=stop_event,
            ):
                decoded_str = base64.b64decode(chunk).decode("utf-8")
                chunk = filter_control_codes(decoded_str)
                if "\n" not in chunk:
                    current += chunk
                    continue
                for i, line in enumerate(chunk.split("\n")):
                    if i == 0:
                        line = current + line
                        current = ""
                    if i == len(chunk.split("\n")) - 1:
                        current = line
                        break
                    console.print(f"[{color}]\\[{pod.runner_name}][/] {line}")
            if current:
                console.print(f"[{color}]\\[{pod.runner_name}][/] {current}")

        heartbeat_thread = Thread(target=heartbeat, args=(stop_event,))
        heartbeat_thread.start()
        workers.append(heartbeat_thread)
        try:
            for pod in pods:
                if pod.labels.get("yatai.ai/is-bento-image-builder") == "true":
                    continue
                thread = Thread(target=pod_log_worker, args=(pod, stop_event))
                thread.start()
                workers.append(thread)
            yield
        finally:
            stop_event.set()
            for thread in workers:
                thread.join()


@attr.define
class DeploymentAPI:
    _client: RestApiClient = attr.field(repr=False)

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

    def _generate_deployment_info_(
        self, res: DeploymentSchema, urls: list[str] | None = None
    ) -> Deployment:
        client = self._client
        route = "deployments"
        if res.manifest and res.manifest.dev:
            route = "codespaces"
        admin_console = f"{client.endpoint}/{route}/{res.name}"
        if res.cluster.is_first is False:
            admin_console = f"{admin_console}?cluster={res.cluster.name}&namespace={res.kube_namespace}"
        return Deployment(
            name=res.name,
            admin_console=admin_console,
            created_at=res.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            created_by=res.creator.name,
            cluster=res.cluster.name,
            _client=self._client,
            _schema=res,
            _urls=urls,
        )

    def list(
        self,
        cluster: str | None = None,
        search: str | None = None,
        dev: bool = False,
        q: str | None = None,
    ) -> list[Deployment]:
        """
        List all deployments in the cluster.

        Args:
            cluster: The name of the cluster to list deployments from. If not specified, list deployments from all clusters.
            search: The search query to filter deployments.

        Returns:
            A list of DeploymentInfo objects.
        """
        cloud_rest_client = self._client
        if cluster is None:
            res_count = cloud_rest_client.v2.list_deployment(all=True, search=search)
            if res_count.total == 0:
                return []
            q = "is:dev" if dev else q
            res = cloud_rest_client.v2.list_deployment(
                search=search, count=res_count.total, all=True, q=q
            )
        else:
            res_count = cloud_rest_client.v2.list_deployment(cluster, search=search)
            if res_count.total == 0:
                return []
            res = cloud_rest_client.v2.list_deployment(
                cluster, search=search, count=res_count.total
            )
        return [self._generate_deployment_info_(schema) for schema in res.items]

    def create(
        self,
        deployment_config_params: DeploymentConfigParameters,
    ) -> Deployment:
        """
        Create a new deployment.

        Args:
            deployment_config_params: The parameters for the deployment.

        Returns:
            The DeploymentInfo object.
        """
        rest_client = self._client
        cluster = deployment_config_params.get_cluster()
        if (
            deployment_config_params.cfg_dict is None
            or deployment_config_params.cfg_dict.get("bento") is None
        ):
            raise ValueError("bento is required")

        config_params = deployment_config_params.get_config_dict()
        config_struct = bentoml_cattr.structure(config_params, CreateDeploymentSchemaV2)
        self._fix_and_validate_schema(config_struct)

        res = rest_client.v2.create_deployment(
            create_schema=config_struct, cluster=cluster
        )

        logger.debug("Deployment Schema: %s", config_struct)

        return self._generate_deployment_info_(res, res.urls)

    def update(
        self,
        deployment_config_params: DeploymentConfigParameters,
    ) -> Deployment:
        """
        Update a deployment.

        Args:
            deployment_config_params: The parameters for the deployment.

        Returns:
            The DeploymentInfo object.
        """
        name = deployment_config_params.get_name()
        if name is None:
            raise ValueError("name is required")
        cluster = deployment_config_params.get_cluster()

        deployment_schema = self._client.v2.get_deployment(name, cluster)
        orig_dict = self._convert_schema_to_update_schema(deployment_schema)

        config_params = deployment_config_params.get_config_dict(
            orig_dict.get("bento"),
        )

        config_merger.merge(orig_dict, config_params)
        config_struct = bentoml_cattr.structure(orig_dict, UpdateDeploymentSchemaV2)

        self._fix_and_validate_schema(config_struct)

        res = self._client.v2.update_deployment(
            cluster=deployment_schema.cluster.name,
            name=name,
            update_schema=config_struct,
        )
        logger.debug("Deployment Schema: %s", config_struct)
        return self._generate_deployment_info_(res, res.urls)

    def void_update(self, name: str, cluster: str | None) -> Deployment:
        res = self._client.v2.void_update_deployment(name, cluster)
        return self._generate_deployment_info_(res, res.urls)

    def apply(
        self,
        deployment_config_params: DeploymentConfigParameters,
    ) -> Deployment:
        """
        Apply a deployment.

        Args:
            deployment_config_params: The parameters for the deployment.

        Returns:
            The DeploymentInfo object.
        """
        name = deployment_config_params.get_name()
        rest_client = self._client
        if name is not None:
            try:
                deployment_schema = rest_client.v2.get_deployment(
                    name, deployment_config_params.get_cluster(pop=False)
                )
            except NotFound:
                return self.create(
                    deployment_config_params=deployment_config_params,
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
            self._fix_and_validate_schema(config_struct)

            res = rest_client.v2.update_deployment(
                name=name,
                update_schema=config_struct,
                cluster=deployment_schema.cluster.name,
            )
            logger.debug("Deployment Schema: %s", config_struct)
            return self._generate_deployment_info_(res, res.urls)

        return self.create(deployment_config_params=deployment_config_params)

    def get(self, name: str, cluster: str | None = None) -> Deployment:
        """
        Get a deployment.

        Args:
            name: The name of the deployment.
            cluster: The name of the cluster.

        Returns:
            The DeploymentInfo object.
        """
        res = self._client.v2.get_deployment(name, cluster)
        return self._generate_deployment_info_(res, res.urls)

    def terminate(
        self, name: str, cluster: str | None = None, wait: bool = False
    ) -> Deployment:
        """
        Terminate a deployment.

        Args:
            name: The name of the deployment.
            cluster: The name of the cluster.
            wait: Whether to wait for the deployment to be terminated.

        Returns:
            The DeploymentInfo object.
        """
        res = self._client.v2.terminate_deployment(name, cluster)
        deployment = self._generate_deployment_info_(res, res.urls)
        if wait:
            console = rich.get_console()
            status = deployment.get_status(False).status
            with console.status(
                f"Waiting for deployment to terminate, current_status: [green]{status}[/]"
            ):
                while status != DeploymentStatus.Terminated.value:
                    time.sleep(1)
                    status = deployment.get_status(True).status
        return deployment

    def delete(self, name: str, cluster: str | None = None) -> None:
        """
        Delete a deployment.

        Args:
            name: The name of the deployment.
            cluster: The name of the cluster.
        """
        self._client.v2.delete_deployment(name, cluster)

    def list_instance_types(self, cluster: str | None = None) -> list[InstanceTypeInfo]:
        """
        List all instance types in the cluster.

        Args:
            cluster: The name of the cluster.

        Returns:
            A list of InstanceTypeInfo objects.
        """
        res = self._client.v2.list_instance_types(cluster)
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


REQUIREMENTS_TXT = "requirements.txt"
EDITABLE_BENTOML_DIR = "__editable_bentoml__"
EDITABLE_BENTOML_PATHSPEC = PathSpec.from_lines(
    "gitwildmatch",
    [
        "/src/",
        "/pyproject.toml",
        "/README.md",
        "/LICENSE",
        "!__pycache__/",
        "!.DS_Store",
    ],
)


def _build_requirements_txt(bento_dir: str) -> bytes:
    from bentoml._internal.configuration import get_bentoml_requirement

    config = BentoBuildConfig.from_bento_dir(bento_dir)
    filename = config.python.requirements_txt
    content = b""
    if filename and os.path.exists(fullpath := os.path.join(bento_dir, filename)):
        with open(fullpath, "rb") as f:
            content = f.read().rstrip(b"\n") + b"\n"
    for package in config.python.packages or []:
        content += f"{package}\n".encode()
    bentoml_requirement = get_bentoml_requirement()
    if bentoml_requirement is None:
        bentoml_requirement = f"-e ./{EDITABLE_BENTOML_DIR}"
    content += f"{bentoml_requirement}\n".encode("utf8")
    return content


def _build_setup_script(bento_dir: str) -> bytes:
    content = b""
    config = BentoBuildConfig.from_bento_dir(bento_dir)
    if config.docker.system_packages:
        content += f"apt-get update && apt-get install -y {' '.join(config.docker.system_packages)} || exit 1\n".encode()
    if config.docker.setup_script and os.path.exists(
        fullpath := os.path.join(bento_dir, config.docker.setup_script)
    ):
        with open(fullpath, "rb") as f:
            content += f.read()
    return content
