import io
import typing as t
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import wraps
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import fs
import requests
from rich.live import Live
from simple_di import inject
from simple_di import Provide
from rich.panel import Panel
from rich.console import Group
from rich.progress import TaskID
from rich.progress import Progress
from rich.progress import BarColumn
from rich.progress import TextColumn
from rich.progress import SpinnerColumn
from rich.progress import DownloadColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn

from ..bento import Bento
from ..bento import BentoStore
from ..bento import SysPathBento
from ..types import Tag
from ..models import Model
from ..models import copy_model
from ..models import ModelStore
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..configuration.containers import BentoMLContainer
from ..yatai_rest_api_client.config import get_current_yatai_rest_api_client
from ..yatai_rest_api_client.schemas import BentoApiSchema
from ..yatai_rest_api_client.schemas import LabelItemSchema
from ..yatai_rest_api_client.schemas import BentoUploadStatus
from ..yatai_rest_api_client.schemas import CreateBentoSchema
from ..yatai_rest_api_client.schemas import CreateModelSchema
from ..yatai_rest_api_client.schemas import ModelUploadStatus
from ..yatai_rest_api_client.schemas import BentoManifestSchema
from ..yatai_rest_api_client.schemas import ModelManifestSchema
from ..yatai_rest_api_client.schemas import FinishUploadBentoSchema
from ..yatai_rest_api_client.schemas import FinishUploadModelSchema
from ..yatai_rest_api_client.schemas import CreateBentoRepositorySchema
from ..yatai_rest_api_client.schemas import CreateModelRepositorySchema


class ObjectWrapper(object):
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        return setattr(self._wrapped, name, value)

    def wrapper_getattr(self, name):
        """Actual `self.getattr` rather than self._wrapped.getattr"""
        try:
            return object.__getattr__(self, name)
        except AttributeError:  # py2
            return getattr(self, name)

    def wrapper_setattr(self, name, value):
        """Actual `self.setattr` rather than self._wrapped.setattr"""
        return object.__setattr__(self, name, value)

    def __init__(self, wrapped):
        """
        Thin wrapper around a given object
        """
        self.wrapper_setattr("_wrapped", wrapped)


class CallbackIOWrapper(ObjectWrapper):
    def __init__(self, callback, stream, method="read"):
        """
        Wrap a given `file`-like object's `read()` or `write()` to report
        lengths to the given `callback`
        """
        super(CallbackIOWrapper, self).__init__(stream)
        func = getattr(stream, method)
        if method == "write":

            @wraps(func)
            def write(data, *args, **kwargs):
                res = func(data, *args, **kwargs)
                callback(len(data))
                return res

            self.wrapper_setattr("write", write)
        elif method == "read":

            @wraps(func)
            def read(*args, **kwargs):
                data = func(*args, **kwargs)
                callback(len(data))
                return data

            self.wrapper_setattr("read", read)
        else:
            raise KeyError("Can only wrap read/write methods")


class YataiClient:
    log_progress = Progress(
        TextColumn("{task.description}"),
    )

    spinner_progress = Progress(
        TextColumn("  "),
        TimeElapsedColumn(),
        TextColumn("[bold purple]{task.fields[action]}"),
        SpinnerColumn("simpleDots"),
    )

    transmission_progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    progress_group = Group(
        Panel(Group(log_progress, spinner_progress)), transmission_progress
    )

    @contextmanager
    def spin(self, *, text: str):
        task_id = self.spinner_progress.add_task("", action=text)
        try:
            yield
        finally:
            self.spinner_progress.stop_task(task_id)
            self.spinner_progress.update(task_id, visible=False)

    @inject
    def push_bento(
        self,
        bento: "Bento",
        *,
        force: bool = False,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        with Live(self.progress_group):
            upload_task_id = self.transmission_progress.add_task(
                f"Pushing bento {bento.tag}"
            )
            self._do_push_bento(
                bento, upload_task_id, force=force, model_store=model_store
            )

    @inject
    def _do_push_bento(
        self,
        bento: "Bento",
        upload_task_id: TaskID,
        *,
        force: bool = False,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        yatai_rest_client = get_current_yatai_rest_api_client()
        name = bento.tag.name
        version = bento.tag.version
        info = bento.info
        model_names = info.models
        with ThreadPoolExecutor(max_workers=max(len(model_names), 1)) as executor:

            def push_model(model: "Model"):
                model_upload_task_id = self.transmission_progress.add_task(
                    f"Pushing model {model.tag}"
                )
                self._do_push_model(model, model_upload_task_id, force=force)

            futures = executor.map(
                push_model, (model_store.get(name) for name in model_names)
            )
            list(futures)
        with self.spin(text=f"Fetching bento repository {name}"):
            bento_repository = yatai_rest_client.get_bento_repository(
                bento_repository_name=name
            )
        if not bento_repository:
            bento_repository = yatai_rest_client.create_bento_repository(
                req=CreateBentoRepositorySchema(name=name, description="")
            )
        with self.spin(text=f"Fetching bento {version}"):
            remote_bento = yatai_rest_client.get_bento(
                bento_repository_name=name, version=version
            )
        if (
            not force
            and remote_bento
            and remote_bento.upload_status == BentoUploadStatus.SUCCESS
        ):
            self.log_progress.add_task(
                f"[bold blue]Bento {bento.tag} already exists in yatai, skipping."
            )
            self.transmission_progress.stop_task(upload_task_id)
            self.transmission_progress.update(upload_task_id, visible=False)
            return
        if not remote_bento:
            labels: t.List[LabelItemSchema] = [
                LabelItemSchema(key=key, value=value)
                for key, value in info.labels.items()
            ]
            apis: t.Dict[str, BentoApiSchema] = {}
            yatai_rest_client.create_bento(
                bento_repository_name=bento_repository.name,
                req=CreateBentoSchema(
                    description="",
                    version=version,
                    build_at=info.creation_time,
                    manifest=BentoManifestSchema(
                        service=info.service,
                        bentoml_version=info.bentoml_version,
                        apis=apis,
                        models=info.models,
                    ),
                    labels=labels,
                ),
            )
        remote_bento = yatai_rest_client.presign_bento_upload_url(
            bento_repository_name=bento_repository.name, version=version
        )
        with io.BytesIO() as tar_io:
            bento_dir_path = bento.path
            with self.spin(text=f"Taring bento {bento.tag}"):
                with tarfile.open(fileobj=tar_io, mode="w:gz") as tar:

                    def filter_(
                        tar_info: tarfile.TarInfo,
                    ) -> t.Optional[tarfile.TarInfo]:
                        if tar_info.path == "./models" or tar_info.path.startswith(
                            "./models/"
                        ):
                            return None
                        return tar_info

                    tar.add(bento_dir_path, arcname="./", filter=filter_)
            tar_io.seek(0, 0)
            with self.spin(text=f"Starting upload bento {bento.tag}"):
                yatai_rest_client.start_upload_bento(
                    bento_repository_name=bento_repository.name, version=version
                )

            file_size = tar_io.getbuffer().nbytes

            self.transmission_progress.update(
                upload_task_id, completed=0, total=file_size
            )
            wrapped_file = CallbackIOWrapper(
                lambda x: self.transmission_progress.update(upload_task_id, advance=x),
                tar_io,
                "read",
            )
            finish_req = FinishUploadBentoSchema(
                status=BentoUploadStatus.SUCCESS,
                reason="",
            )
            try:
                resp = requests.put(
                    remote_bento.presigned_upload_url, data=wrapped_file
                )
                if resp.status_code != 200:
                    finish_req = FinishUploadBentoSchema(
                        status=BentoUploadStatus.FAILED,
                        reason=resp.text,
                    )
            except Exception as e:
                finish_req = FinishUploadBentoSchema(
                    status=BentoUploadStatus.FAILED,
                    reason=str(e),
                )
            yatai_rest_client.finish_upload_bento(
                bento_repository_name=bento_repository.name,
                version=version,
                req=finish_req,
            )
            if finish_req.status != BentoUploadStatus.SUCCESS:
                self.log_progress.add_task(
                    f"[bold red]Upload bento {bento.tag} failed: {finish_req.reason}"
                )
            else:
                self.log_progress.add_task(
                    f"[bold green]Upload bento {bento.tag} successfully"
                )

    @inject
    def pull_bento(
        self,
        tag: t.Union[str, Tag],
        *,
        force: bool = False,
        bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "Bento":
        with Live(self.progress_group):
            download_task_id = self.transmission_progress.add_task(
                f"Pulling bento {tag}"
            )
            return self._do_pull_bento(
                tag,
                download_task_id,
                force=force,
                bento_store=bento_store,
                model_store=model_store,
            )

    @inject
    def _do_pull_bento(
        self,
        tag: t.Union[str, Tag],
        download_task_id: TaskID,
        *,
        force: bool = False,
        bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "Bento":
        try:
            bento = bento_store.get(tag)
            if not force:
                self.log_progress.add_task(
                    f"[bold blue]Bento {tag} already exists locally, skipping pull"
                )
                self.transmission_progress.stop_task(download_task_id)
                self.transmission_progress.update(download_task_id, visible=False)
                return bento
            bento_store.delete(tag)
        except NotFound:
            pass
        _tag = Tag.from_taglike(tag)
        yatai_rest_client = get_current_yatai_rest_api_client()
        remote_bento = yatai_rest_client.get_bento(
            bento_repository_name=_tag.name, version=_tag.version
        )
        with ThreadPoolExecutor(
            max_workers=max(len(remote_bento.manifest.models), 1)
        ) as executor:

            def pull_model(model_tag: Tag):
                model_download_task_id = self.transmission_progress.add_task(
                    f"Pulling model {model_tag}"
                )
                self._do_pull_model(
                    model_tag,
                    model_download_task_id,
                    force=force,
                    model_store=model_store,
                )

            futures = executor.map(pull_model, remote_bento.manifest.models)
            list(futures)
        remote_bento = yatai_rest_client.presign_bento_download_url(
            _tag.name, _tag.version
        )
        url = remote_bento.presigned_download_url
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise BentoMLException(f"Failed to download bento {_tag}: {response.text}")
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        with NamedTemporaryFile() as tar_file:
            self.transmission_progress.update(
                download_task_id, completed=0, total=total_size_in_bytes
            )
            for data in response.iter_content(block_size):
                self.transmission_progress.update(download_task_id, advance=len(data))
                tar_file.write(data)
            self.log_progress.add_task(
                f"[bold green]Download bento {_tag} successfully"
            )
            tar_file.seek(0, 0)
            tar = tarfile.open(fileobj=tar_file, mode="r:gz")
            with fs.open_fs("temp://") as temp_fs:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    p = Path(member.name)
                    if p.parent != Path("."):
                        temp_fs.makedirs(str(p.parent), recreate=True)
                    temp_fs.writebytes(member.name, f.read())
                bento = SysPathBento.from_Bento(Bento.from_fs(temp_fs)).save(
                    bento_store
                )
                for model_tag in remote_bento.manifest.models:
                    with self.spin(text=f"Copying model {model_tag}"):
                        copy_model(
                            model_tag,
                            src_model_store=model_store,
                            target_model_store=bento._model_store,
                        )
                return bento

    def push_model(self, model: "Model", *, force: bool = False):
        with Live(self.progress_group):
            upload_task_id = self.transmission_progress.add_task(
                f"Pushing model {model.tag}"
            )
            self._do_push_model(model, upload_task_id, force=force)

    def _do_push_model(
        self, model: "Model", upload_task_id: TaskID, *, force: bool = False
    ):
        yatai_rest_client = get_current_yatai_rest_api_client()
        name = model.tag.name
        version = model.tag.version
        info = model.info
        with self.spin(text=f"Fetching model {model.tag}"):
            model_repository = yatai_rest_client.get_model_repository(
                model_repository_name=name
            )
        if not model_repository:
            model_repository = yatai_rest_client.create_model_repository(
                req=CreateModelRepositorySchema(name=name, description="")
            )
        with self.spin(text=f"Fetching model version {version}"):
            remote_model = yatai_rest_client.get_model(
                model_repository_name=name, version=version
            )
        if (
            not force
            and remote_model
            and remote_model.upload_status == ModelUploadStatus.SUCCESS
        ):
            self.log_progress.add_task(
                f"[bold blue]Model {model.tag} already exists in yatai, skipping."
            )
            self.transmission_progress.stop_task(upload_task_id)
            self.transmission_progress.update(upload_task_id, visible=False)
            return
        if not remote_model:
            labels: t.List[LabelItemSchema] = [
                LabelItemSchema(key=key, value=value)
                for key, value in info.labels.items()
            ]
            yatai_rest_client.create_model(
                model_repository_name=model_repository.name,
                req=CreateModelSchema(
                    description="",
                    version=version,
                    build_at=info.creation_time,
                    manifest=ModelManifestSchema(
                        module=info.module,
                        metadata=info.metadata,
                        context=info.context,
                        options=info.options,
                        api_version=info.api_version,
                        bentoml_version=info.bentoml_version,
                    ),
                    labels=labels,
                ),
            )
        remote_model = yatai_rest_client.presign_model_upload_url(
            model_repository_name=model_repository.name, version=version
        )
        with io.BytesIO() as tar_io:
            bento_dir_path = model.path
            with self.spin(text=f"Taring model {model.tag}"):
                with tarfile.open(fileobj=tar_io, mode="w:gz") as tar:
                    tar.add(bento_dir_path, arcname="./")
            tar_io.seek(0, 0)
            with self.spin(text=f"Starting upload model {model.tag}"):
                yatai_rest_client.start_upload_model(
                    model_repository_name=model_repository.name, version=version
                )
            file_size = tar_io.getbuffer().nbytes
            self.transmission_progress.update(
                upload_task_id,
                description=f"Pushing model {model.tag}",
                total=file_size,
            )
            wrapped_file = CallbackIOWrapper(
                lambda x: self.transmission_progress.update(upload_task_id, advance=x),
                tar_io,
                "read",
            )
            finish_req = FinishUploadModelSchema(
                status=ModelUploadStatus.SUCCESS,
                reason="",
            )
            try:
                resp = requests.put(
                    remote_model.presigned_upload_url, data=wrapped_file
                )
                if resp.status_code != 200:
                    finish_req = FinishUploadModelSchema(
                        status=ModelUploadStatus.FAILED,
                        reason=resp.text,
                    )
            except Exception as e:
                finish_req = FinishUploadModelSchema(
                    status=ModelUploadStatus.FAILED,
                    reason=str(e),
                )
            yatai_rest_client.finish_upload_model(
                model_repository_name=model_repository.name,
                version=version,
                req=finish_req,
            )
            if finish_req.status != ModelUploadStatus.SUCCESS:
                self.log_progress.add_task(
                    f"[bold red]Upload model {model.tag} failed: {finish_req.reason}"
                )
            else:
                self.log_progress.add_task(
                    f"[bold green]Upload model {model.tag} successfully"
                )

    @inject
    def pull_model(
        self,
        tag: t.Union[str, Tag],
        *,
        force: bool = False,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "Model":
        with Live(self.progress_group):
            download_task_id = self.transmission_progress.add_task(
                f"Pulling model {tag}"
            )
            return self._do_pull_model(
                tag, download_task_id, force=force, model_store=model_store
            )

    @inject
    def _do_pull_model(
        self,
        tag: t.Union[str, Tag],
        download_task_id: TaskID,
        *,
        force: bool = False,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "Model":
        try:
            model = model_store.get(tag)
            if not force:
                self.log_progress.add_task(
                    f"[bold blue]Model {tag} already exists locally, skipping pull"
                )
                self.transmission_progress.stop_task(download_task_id)
                self.transmission_progress.update(download_task_id, visible=False)
                return model
            model_store.delete(tag)
        except NotFound:
            pass
        yatai_rest_client = get_current_yatai_rest_api_client()
        _tag = Tag.from_taglike(tag)
        remote_model = yatai_rest_client.presign_model_download_url(
            _tag.name, _tag.version
        )
        url = remote_model.presigned_download_url
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise BentoMLException(f"Failed to download model {_tag}: {response.text}")
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        with NamedTemporaryFile() as tar_file:
            self.transmission_progress.update(
                download_task_id,
                description=f"Pulling model {_tag}",
                total=total_size_in_bytes,
            )
            for data in response.iter_content(block_size):
                self.transmission_progress.update(download_task_id, advance=len(data))
                tar_file.write(data)
            self.log_progress.add_task(
                f"[bold green]Download model {_tag} successfully"
            )
            tar_file.seek(0, 0)
            tar = tarfile.open(fileobj=tar_file, mode="r:gz")
            with fs.open_fs("temp://") as temp_fs:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    p = Path(member.name)
                    if p.parent != Path("."):
                        temp_fs.makedirs(str(p.parent), recreate=True)
                    temp_fs.writebytes(member.name, f.read())
                return Model.from_fs(temp_fs).save(model_store)


yatai_client = YataiClient()
