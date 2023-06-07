from __future__ import annotations

import io
import typing as t
import tarfile
import tempfile
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor

import fs
import requests
from rich.live import Live
from simple_di import inject
from simple_di import Provide

from ..tag import Tag
from ..bento import Bento
from ..bento import BentoStore
from ..utils import calc_dir_size
from .config import get_rest_api_client
from ..models import Model
from ..models import copy_model
from ..models import ModelStore
from .schemas import BentoApiSchema
from .schemas import LabelItemSchema
from .schemas import BentoRunnerSchema
from .schemas import BentoUploadStatus
from .schemas import CreateBentoSchema
from .schemas import CreateModelSchema
from .schemas import ModelUploadStatus
from .schemas import UpdateBentoSchema
from .schemas import CompletePartSchema
from .schemas import BentoManifestSchema
from .schemas import ModelManifestSchema
from .schemas import TransmissionStrategy
from .schemas import FinishUploadBentoSchema
from .schemas import FinishUploadModelSchema
from .schemas import BentoRunnerResourceSchema
from .schemas import CreateBentoRepositorySchema
from .schemas import CreateModelRepositorySchema
from .schemas import CompleteMultipartUploadSchema
from .schemas import PreSignMultipartUploadUrlSchema
from .basecloud import BaseCloudClient
from .basecloud import FILE_CHUNK_SIZE
from .basecloud import CallbackIOWrapper
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from rich.progress import TaskID


class YataiClient(BaseCloudClient):
    def push_bento(
        self,
        bento: Bento,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ):
        with Live(self.progress_group):
            upload_task_id = self.transmission_progress.add_task(
                f'Pushing Bento "{bento.tag}"', start=False, visible=False
            )
            self._do_push_bento(
                bento, upload_task_id, force=force, threads=threads, context=context
            )

    def _do_push_bento(
        self,
        bento: Bento,
        upload_task_id: TaskID,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ):
        yatai_rest_client = get_rest_api_client(context)
        name = bento.tag.name
        version = bento.tag.version
        if version is None:
            raise BentoMLException(f"Bento {bento.tag} version cannot be None")
        info = bento.info
        model_tags = [m.tag for m in info.models]
        model_store = bento._model_store  # type: ignore
        models = (model_store.get(name) for name in model_tags)
        with ThreadPoolExecutor(max_workers=max(len(model_tags), 1)) as executor:

            def push_model(model: Model) -> None:
                model_upload_task_id = self.transmission_progress.add_task(
                    f'Pushing model "{model.tag}"', start=False, visible=False
                )
                self._do_push_model(
                    model,
                    model_upload_task_id,
                    force=force,
                    threads=threads,
                    context=context,
                )

            futures: t.Iterator[None] = executor.map(push_model, models)
            list(futures)
        with self.spin(text=f'Fetching Bento repository "{name}"'):
            bento_repository = yatai_rest_client.get_bento_repository(
                bento_repository_name=name
            )
        if not bento_repository:
            with self.spin(text=f'Bento repository "{name}" not found, creating now..'):
                bento_repository = yatai_rest_client.create_bento_repository(
                    req=CreateBentoRepositorySchema(name=name, description="")
                )
        with self.spin(text=f'Try fetching Bento "{bento.tag}" from Yatai..'):
            remote_bento = yatai_rest_client.get_bento(
                bento_repository_name=name, version=version
            )
        if (
            not force
            and remote_bento
            and remote_bento.upload_status == BentoUploadStatus.SUCCESS
        ):
            self.log_progress.add_task(
                f'[bold blue]Push failed: Bento "{bento.tag}" already exists in Yatai'
            )
            return
        labels: list[LabelItemSchema] = [
            LabelItemSchema(key=key, value=value) for key, value in info.labels.items()
        ]
        apis: dict[str, BentoApiSchema] = {}
        models = [str(m.tag) for m in info.models]
        runners = [
            BentoRunnerSchema(
                name=r.name,
                runnable_type=r.runnable_type,
                models=r.models,
                resource_config=BentoRunnerResourceSchema(
                    cpu=r.resource_config.get("cpu"),
                    nvidia_gpu=r.resource_config.get("nvidia.com/gpu"),
                    custom_resources=r.resource_config.get("custom_resources"),
                )
                if r.resource_config
                else None,
            )
            for r in info.runners
        ]
        manifest = BentoManifestSchema(
            service=info.service,
            bentoml_version=info.bentoml_version,
            apis=apis,
            models=models,
            runners=runners,
            size_bytes=calc_dir_size(bento.path),
        )
        if not remote_bento:
            with self.spin(text=f'Registering Bento "{bento.tag}" with Yatai..'):
                remote_bento = yatai_rest_client.create_bento(
                    bento_repository_name=bento_repository.name,
                    req=CreateBentoSchema(
                        description="",
                        version=version,
                        build_at=info.creation_time,
                        manifest=manifest,
                        labels=labels,
                    ),
                )
        else:
            with self.spin(text=f'Updating Bento "{bento.tag}"..'):
                remote_bento = yatai_rest_client.update_bento(
                    bento_repository_name=bento_repository.name,
                    version=version,
                    req=UpdateBentoSchema(
                        manifest=manifest,
                        labels=labels,
                    ),
                )

        transmission_strategy: TransmissionStrategy = "proxy"
        presigned_upload_url: str | None = None

        if remote_bento.transmission_strategy is not None:
            transmission_strategy = remote_bento.transmission_strategy
        else:
            with self.spin(
                text=f'Getting a presigned upload url for bento "{bento.tag}" ..'
            ):
                remote_bento = yatai_rest_client.presign_bento_upload_url(
                    bento_repository_name=bento_repository.name, version=version
                )
                if remote_bento.presigned_upload_url:
                    transmission_strategy = "presigned_url"
                    presigned_upload_url = remote_bento.presigned_upload_url

        with io.BytesIO() as tar_io:
            with self.spin(text=f'Creating tar archive for bento "{bento.tag}"..'):
                with tarfile.open(fileobj=tar_io, mode="w:gz") as tar:

                    def filter_(
                        tar_info: tarfile.TarInfo,
                    ) -> t.Optional[tarfile.TarInfo]:
                        if tar_info.path == "./models" or tar_info.path.startswith(
                            "./models/"
                        ):
                            return None
                        return tar_info

                    tar.add(bento.path, arcname="./", filter=filter_)
            tar_io.seek(0, 0)

            with self.spin(text=f'Start uploading bento "{bento.tag}"..'):
                yatai_rest_client.start_upload_bento(
                    bento_repository_name=bento_repository.name, version=version
                )

            file_size = tar_io.getbuffer().nbytes

            self.transmission_progress.update(
                upload_task_id, completed=0, total=file_size, visible=True
            )
            self.transmission_progress.start_task(upload_task_id)

            io_mutex = threading.Lock()

            def io_cb(x: int):
                with io_mutex:
                    self.transmission_progress.update(upload_task_id, advance=x)

            wrapped_file = CallbackIOWrapper(io_cb, tar_io, "read")

            if transmission_strategy == "proxy":
                try:
                    yatai_rest_client.upload_bento(
                        bento_repository_name=bento_repository.name,
                        version=version,
                        data=wrapped_file,
                    )
                except Exception as e:  # pylint: disable=broad-except
                    self.log_progress.add_task(
                        f'[bold red]Failed to upload bento "{bento.tag}"'
                    )
                    raise e
                self.log_progress.add_task(
                    f'[bold green]Successfully pushed bento "{bento.tag}"'
                )
                return
            finish_req = FinishUploadBentoSchema(
                status=BentoUploadStatus.SUCCESS,
                reason="",
            )
            try:
                if presigned_upload_url is not None:
                    resp = requests.put(presigned_upload_url, data=wrapped_file)
                    if resp.status_code != 200:
                        finish_req = FinishUploadBentoSchema(
                            status=BentoUploadStatus.FAILED,
                            reason=resp.text,
                        )
                else:
                    with self.spin(
                        text=f'Start multipart uploading Bento "{bento.tag}"...'
                    ):
                        remote_bento = yatai_rest_client.start_bento_multipart_upload(
                            bento_repository_name=bento_repository.name,
                            version=version,
                        )
                        if not remote_bento.upload_id:
                            raise BentoMLException(
                                f'Failed to start multipart upload for Bento "{bento.tag}", upload_id is empty'
                            )

                        upload_id: str = remote_bento.upload_id

                    chunks_count = file_size // FILE_CHUNK_SIZE + 1

                    def chunk_upload(
                        upload_id: str, chunk_number: int
                    ) -> FinishUploadBentoSchema | tuple[str, int]:
                        with self.spin(
                            text=f'({chunk_number}/{chunks_count}) Presign multipart upload url of Bento "{bento.tag}"...'
                        ):
                            remote_bento = (
                                yatai_rest_client.presign_bento_multipart_upload_url(
                                    bento_repository_name=bento_repository.name,
                                    version=version,
                                    req=PreSignMultipartUploadUrlSchema(
                                        upload_id=upload_id,
                                        part_number=chunk_number,
                                    ),
                                )
                            )
                        with self.spin(
                            text=f'({chunk_number}/{chunks_count}) Uploading chunk of Bento "{bento.tag}"...'
                        ):
                            chunk = (
                                tar_io.getbuffer()[
                                    (chunk_number - 1)
                                    * FILE_CHUNK_SIZE : chunk_number
                                    * FILE_CHUNK_SIZE
                                ]
                                if chunk_number < chunks_count
                                else tar_io.getbuffer()[
                                    (chunk_number - 1) * FILE_CHUNK_SIZE :
                                ]
                            )

                            with io.BytesIO(chunk) as chunk_io:
                                wrapped_file = CallbackIOWrapper(
                                    io_cb, chunk_io, "read"
                                )

                                resp = requests.put(
                                    remote_bento.presigned_upload_url, data=wrapped_file
                                )
                                if resp.status_code != 200:
                                    return FinishUploadBentoSchema(
                                        status=BentoUploadStatus.FAILED,
                                        reason=resp.text,
                                    )
                                return resp.headers["ETag"], chunk_number

                    futures_: list[
                        Future[FinishUploadBentoSchema | tuple[str, int]]
                    ] = []

                    with ThreadPoolExecutor(
                        max_workers=min(max(chunks_count, 1), threads)
                    ) as executor:
                        for i in range(1, chunks_count + 1):
                            future = executor.submit(
                                chunk_upload,
                                upload_id,
                                i,
                            )
                            futures_.append(future)

                    parts: list[CompletePartSchema] = []

                    for future in futures_:
                        result = future.result()
                        if isinstance(result, FinishUploadBentoSchema):
                            finish_req = result
                            break
                        else:
                            etag, chunk_number = result
                            parts.append(
                                CompletePartSchema(
                                    part_number=chunk_number,
                                    etag=etag,
                                )
                            )

                    with self.spin(
                        text=f'Completing multipart upload of Bento "{bento.tag}"...'
                    ):
                        remote_bento = (
                            yatai_rest_client.complete_bento_multipart_upload(
                                bento_repository_name=bento_repository.name,
                                version=version,
                                req=CompleteMultipartUploadSchema(
                                    upload_id=upload_id,
                                    parts=parts,
                                ),
                            )
                        )

            except Exception as e:  # pylint: disable=broad-except
                finish_req = FinishUploadBentoSchema(
                    status=BentoUploadStatus.FAILED,
                    reason=str(e),
                )
            if finish_req.status is BentoUploadStatus.FAILED:
                self.log_progress.add_task(
                    f'[bold red]Failed to upload Bento "{bento.tag}"'
                )
            with self.spin(text="Submitting upload status to Yatai"):
                yatai_rest_client.finish_upload_bento(
                    bento_repository_name=bento_repository.name,
                    version=version,
                    req=finish_req,
                )
            if finish_req.status != BentoUploadStatus.SUCCESS:
                self.log_progress.add_task(
                    f'[bold red]Failed pushing Bento "{bento.tag}": {finish_req.reason}'
                )
            else:
                self.log_progress.add_task(
                    f'[bold green]Successfully pushed Bento "{bento.tag}"'
                )

    @inject
    def pull_bento(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
        bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
        context: str | None = None,
    ) -> Bento:
        with Live(self.progress_group):
            download_task_id = self.transmission_progress.add_task(
                f'Pulling bento "{tag}"', start=False, visible=False
            )
            return self._do_pull_bento(
                tag,
                download_task_id,
                force=force,
                bento_store=bento_store,
                context=context,
            )

    @inject
    def _do_pull_bento(
        self,
        tag: str | Tag,
        download_task_id: TaskID,
        *,
        force: bool = False,
        bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
        context: str | None = None,
    ) -> Bento:
        try:
            bento = bento_store.get(tag)
            if not force:
                self.log_progress.add_task(
                    f'[bold blue]Bento "{tag}" exists in local model store'
                )
                return bento
            bento_store.delete(tag)
        except NotFound:
            pass
        _tag = Tag.from_taglike(tag)
        name = _tag.name
        version = _tag.version
        if version is None:
            raise BentoMLException(f'Bento "{_tag}" version can not be None')

        yatai_rest_client = get_rest_api_client(context)

        with self.spin(text=f'Fetching bento "{_tag}"'):
            remote_bento = yatai_rest_client.get_bento(
                bento_repository_name=name, version=version
            )
        if not remote_bento:
            raise BentoMLException(f'Bento "{_tag}" not found on Yatai')

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download models to a temporary directory
            model_store = ModelStore(temp_dir)
            with ThreadPoolExecutor(
                max_workers=max(len(remote_bento.manifest.models), 1)
            ) as executor:

                def pull_model(model_tag: Tag):
                    model_download_task_id = self.transmission_progress.add_task(
                        f'Pulling model "{model_tag}"', start=False, visible=False
                    )
                    self._do_pull_model(
                        model_tag,
                        model_download_task_id,
                        force=force,
                        model_store=model_store,
                        context=context,
                    )

                futures = executor.map(pull_model, remote_bento.manifest.models)
                list(futures)

            # Download bento files from yatai
            transmission_strategy: TransmissionStrategy = "proxy"
            presigned_download_url: str | None = None

            if remote_bento.transmission_strategy is not None:
                transmission_strategy = remote_bento.transmission_strategy
            else:
                with self.spin(
                    text=f'Getting a presigned download url for bento "{_tag}"'
                ):
                    remote_bento = yatai_rest_client.presign_bento_download_url(
                        name, version
                    )
                    if remote_bento.presigned_download_url:
                        presigned_download_url = remote_bento.presigned_download_url
                        transmission_strategy = "presigned_url"

            if transmission_strategy == "proxy":
                response = yatai_rest_client.download_bento(
                    bento_repository_name=name,
                    version=version,
                )
            else:
                if presigned_download_url is None:
                    with self.spin(
                        text=f'Getting a presigned download url for bento "{_tag}"'
                    ):
                        remote_bento = yatai_rest_client.presign_bento_download_url(
                            name, version
                        )
                        presigned_download_url = remote_bento.presigned_download_url
                response = requests.get(presigned_download_url, stream=True)

            if response.status_code != 200:
                raise BentoMLException(
                    f'Failed to download bento "{_tag}": {response.text}'
                )
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            with NamedTemporaryFile() as tar_file:
                self.transmission_progress.update(
                    download_task_id,
                    completed=0,
                    total=total_size_in_bytes,
                    visible=True,
                )
                self.transmission_progress.start_task(download_task_id)
                for data in response.iter_content(block_size):
                    self.transmission_progress.update(
                        download_task_id, advance=len(data)
                    )
                    tar_file.write(data)
                self.log_progress.add_task(
                    f'[bold green]Finished downloading all bento "{_tag}" files'
                )
                tar_file.seek(0, 0)
                tar = tarfile.open(fileobj=tar_file, mode="r:gz")
                with self.spin(text=f'Extracting bento "{_tag}" tar file'):
                    with fs.open_fs("temp://") as temp_fs:
                        for member in tar.getmembers():
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            p = Path(member.name)
                            if p.parent != Path("."):
                                temp_fs.makedirs(str(p.parent), recreate=True)
                            temp_fs.writebytes(member.name, f.read())
                        bento = Bento.from_fs(temp_fs)
                        for model_tag in remote_bento.manifest.models:
                            with self.spin(
                                text=f'Copying model "{model_tag}" to bento'
                            ):
                                copy_model(
                                    model_tag,
                                    src_model_store=model_store,
                                    target_model_store=bento._model_store,  # type: ignore
                                )
                        bento = bento.save(bento_store)
                        self.log_progress.add_task(
                            f'[bold green]Successfully pulled bento "{_tag}"'
                        )
                        return bento

    def push_model(
        self,
        model: Model,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ):
        with Live(self.progress_group):
            upload_task_id = self.transmission_progress.add_task(
                f'Pushing model "{model.tag}"', start=False, visible=False
            )
            self._do_push_model(
                model, upload_task_id, force=force, threads=threads, context=context
            )

    def _do_push_model(
        self,
        model: Model,
        upload_task_id: TaskID,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ):
        yatai_rest_client = get_rest_api_client(context)
        name = model.tag.name
        version = model.tag.version
        if version is None:
            raise BentoMLException(f'Model "{model.tag}" version cannot be None')
        info = model.info
        with self.spin(text=f'Fetching model repository "{name}"'):
            model_repository = yatai_rest_client.get_model_repository(
                model_repository_name=name
            )
        if not model_repository:
            with self.spin(text=f'Model repository "{name}" not found, creating now..'):
                model_repository = yatai_rest_client.create_model_repository(
                    req=CreateModelRepositorySchema(name=name, description="")
                )
        with self.spin(text=f'Try fetching model "{model.tag}" from Yatai..'):
            remote_model = yatai_rest_client.get_model(
                model_repository_name=name, version=version
            )
        if (
            not force
            and remote_model
            and remote_model.upload_status == ModelUploadStatus.SUCCESS
        ):
            self.log_progress.add_task(
                f'[bold blue]Model "{model.tag}" already exists in Yatai, skipping'
            )
            return
        if not remote_model:
            labels: list[LabelItemSchema] = [
                LabelItemSchema(key=key, value=value)
                for key, value in info.labels.items()
            ]
            with self.spin(text=f'Registering model "{model.tag}" with Yatai..'):
                remote_model = yatai_rest_client.create_model(
                    model_repository_name=model_repository.name,
                    req=CreateModelSchema(
                        description="",
                        version=version,
                        build_at=info.creation_time,
                        manifest=ModelManifestSchema(
                            module=info.module,
                            metadata=info.metadata,
                            context=info.context.to_dict(),
                            options=info.options.to_dict(),
                            api_version=info.api_version,
                            bentoml_version=info.context.bentoml_version,
                            size_bytes=calc_dir_size(model.path),
                        ),
                        labels=labels,
                    ),
                )

        transmission_strategy: TransmissionStrategy = "proxy"
        presigned_upload_url: str | None = None

        if remote_model.transmission_strategy is not None:
            transmission_strategy = remote_model.transmission_strategy
        else:
            with self.spin(
                text=f'Getting a presigned upload url for Model "{model.tag}" ..'
            ):
                remote_model = yatai_rest_client.presign_model_upload_url(
                    model_repository_name=model_repository.name, version=version
                )
                if remote_model.presigned_upload_url:
                    transmission_strategy = "presigned_url"
                    presigned_upload_url = remote_model.presigned_upload_url

        with io.BytesIO() as tar_io:
            with self.spin(text=f'Creating tar archive for model "{model.tag}"..'):
                with tarfile.open(fileobj=tar_io, mode="w:gz") as tar:
                    tar.add(model.path, arcname="./")
            tar_io.seek(0, 0)
            with self.spin(text=f'Start uploading model "{model.tag}"..'):
                yatai_rest_client.start_upload_model(
                    model_repository_name=model_repository.name, version=version
                )
            file_size = tar_io.getbuffer().nbytes
            self.transmission_progress.update(
                upload_task_id,
                description=f'Uploading model "{model.tag}"',
                total=file_size,
                visible=True,
            )
            self.transmission_progress.start_task(upload_task_id)

            io_mutex = threading.Lock()

            def io_cb(x: int):
                with io_mutex:
                    self.transmission_progress.update(upload_task_id, advance=x)

            wrapped_file = CallbackIOWrapper(io_cb, tar_io, "read")
            if transmission_strategy == "proxy":
                try:
                    yatai_rest_client.upload_model(
                        model_repository_name=model_repository.name,
                        version=version,
                        data=wrapped_file,
                    )
                except Exception as e:  # pylint: disable=broad-except
                    self.log_progress.add_task(
                        f'[bold red]Failed to upload model "{model.tag}"'
                    )
                    raise e
                self.log_progress.add_task(
                    f'[bold green]Successfully pushed model "{model.tag}"'
                )
                return
            finish_req = FinishUploadModelSchema(
                status=ModelUploadStatus.SUCCESS,
                reason="",
            )
            try:
                if presigned_upload_url is not None:
                    resp = requests.put(presigned_upload_url, data=wrapped_file)
                    if resp.status_code != 200:
                        finish_req = FinishUploadModelSchema(
                            status=ModelUploadStatus.FAILED,
                            reason=resp.text,
                        )
                else:
                    with self.spin(
                        text=f'Start multipart uploading Model "{model.tag}"...'
                    ):
                        remote_model = yatai_rest_client.start_model_multipart_upload(
                            model_repository_name=model_repository.name,
                            version=version,
                        )
                        if not remote_model.upload_id:
                            raise BentoMLException(
                                f'Failed to start multipart upload for model "{model.tag}", upload_id is empty'
                            )

                        upload_id: str = remote_model.upload_id

                    chunks_count = file_size // FILE_CHUNK_SIZE + 1

                    def chunk_upload(
                        upload_id: str, chunk_number: int
                    ) -> FinishUploadModelSchema | tuple[str, int]:
                        with self.spin(
                            text=f'({chunk_number}/{chunks_count}) Presign multipart upload url of model "{model.tag}"...'
                        ):
                            remote_model = (
                                yatai_rest_client.presign_model_multipart_upload_url(
                                    model_repository_name=model_repository.name,
                                    version=version,
                                    req=PreSignMultipartUploadUrlSchema(
                                        upload_id=upload_id,
                                        part_number=chunk_number,
                                    ),
                                )
                            )

                        with self.spin(
                            text=f'({chunk_number}/{chunks_count}) Uploading chunk of model "{model.tag}"...'
                        ):
                            chunk = (
                                tar_io.getbuffer()[
                                    (chunk_number - 1)
                                    * FILE_CHUNK_SIZE : chunk_number
                                    * FILE_CHUNK_SIZE
                                ]
                                if chunk_number < chunks_count
                                else tar_io.getbuffer()[
                                    (chunk_number - 1) * FILE_CHUNK_SIZE :
                                ]
                            )

                            with io.BytesIO(chunk) as chunk_io:
                                wrapped_file = CallbackIOWrapper(
                                    io_cb,
                                    chunk_io,
                                    "read",
                                )

                                resp = requests.put(
                                    remote_model.presigned_upload_url, data=wrapped_file
                                )
                                if resp.status_code != 200:
                                    return FinishUploadModelSchema(
                                        status=ModelUploadStatus.FAILED,
                                        reason=resp.text,
                                    )
                                return resp.headers["ETag"], chunk_number

                    futures_: list[
                        Future[FinishUploadModelSchema | tuple[str, int]]
                    ] = []

                    with ThreadPoolExecutor(
                        max_workers=min(max(chunks_count, 1), threads)
                    ) as executor:
                        for i in range(1, chunks_count + 1):
                            future = executor.submit(
                                chunk_upload,
                                upload_id,
                                i,
                            )
                            futures_.append(future)

                    parts: list[CompletePartSchema] = []

                    for future in futures_:
                        result = future.result()
                        if isinstance(result, FinishUploadModelSchema):
                            finish_req = result
                            break
                        else:
                            etag, chunk_number = result
                            parts.append(
                                CompletePartSchema(
                                    part_number=chunk_number,
                                    etag=etag,
                                )
                            )

                    with self.spin(
                        text=f'Completing multipart upload of model "{model.tag}"...'
                    ):
                        remote_model = (
                            yatai_rest_client.complete_model_multipart_upload(
                                model_repository_name=model_repository.name,
                                version=version,
                                req=CompleteMultipartUploadSchema(
                                    upload_id=upload_id,
                                    parts=parts,
                                ),
                            )
                        )

            except Exception as e:  # pylint: disable=broad-except
                finish_req = FinishUploadModelSchema(
                    status=ModelUploadStatus.FAILED,
                    reason=str(e),
                )
            if finish_req.status is ModelUploadStatus.FAILED:
                self.log_progress.add_task(
                    f'[bold red]Failed to upload model "{model.tag}"'
                )
            with self.spin(text="Submitting upload status to Yatai"):
                yatai_rest_client.finish_upload_model(
                    model_repository_name=model_repository.name,
                    version=version,
                    req=finish_req,
                )
            if finish_req.status != ModelUploadStatus.SUCCESS:
                self.log_progress.add_task(
                    f'[bold red]Failed pushing model "{model.tag}" : {finish_req.reason}'
                )
            else:
                self.log_progress.add_task(
                    f'[bold green]Successfully pushed model "{model.tag}"'
                )

    @inject
    def pull_model(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
        context: str | None = None,
        model_store: ModelStore = Provide[BentoMLContainer.model_store],
    ) -> Model:
        with Live(self.progress_group):
            download_task_id = self.transmission_progress.add_task(
                f'Pulling model "{tag}"', start=False, visible=False
            )
            return self._do_pull_model(
                tag,
                download_task_id,
                force=force,
                model_store=model_store,
                context=context,
            )

    @inject
    def _do_pull_model(
        self,
        tag: str | Tag,
        download_task_id: TaskID,
        *,
        force: bool = False,
        model_store: ModelStore = Provide[BentoMLContainer.model_store],
        context: str | None = None,
    ) -> Model:
        try:
            model = model_store.get(tag)
            if not force:
                self.log_progress.add_task(
                    f'[bold blue]Model "{tag}" already exists locally, skipping'
                )
                return model
            else:
                model_store.delete(tag)
        except NotFound:
            pass
        yatai_rest_client = get_rest_api_client(context)
        _tag = Tag.from_taglike(tag)
        name = _tag.name
        version = _tag.version
        if version is None:
            raise BentoMLException(f'Model "{_tag}" version cannot be None')
        with self.spin(text=f'Getting a presigned download url for model "{_tag}"..'):
            remote_model = yatai_rest_client.presign_model_download_url(name, version)

        if not remote_model:
            raise BentoMLException(f'Model "{_tag}" not found on Yatai')

        # Download model files from yatai
        transmission_strategy: TransmissionStrategy = "proxy"
        presigned_download_url: str | None = None

        if remote_model.transmission_strategy is not None:
            transmission_strategy = remote_model.transmission_strategy
        else:
            with self.spin(text=f'Getting a presigned download url for model "{_tag}"'):
                remote_model = yatai_rest_client.presign_model_download_url(
                    name, version
                )
                if remote_model.presigned_download_url:
                    presigned_download_url = remote_model.presigned_download_url
                    transmission_strategy = "presigned_url"

        if transmission_strategy == "proxy":
            response = yatai_rest_client.download_model(
                model_repository_name=name, version=version
            )
        else:
            if presigned_download_url is None:
                with self.spin(
                    text=f'Getting a presigned download url for model "{_tag}"'
                ):
                    remote_model = yatai_rest_client.presign_model_download_url(
                        name, version
                    )
                    presigned_download_url = remote_model.presigned_download_url

            response = requests.get(presigned_download_url, stream=True)
            if response.status_code != 200:
                raise BentoMLException(
                    f'Failed to download model "{_tag}": {response.text}'
                )

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        with NamedTemporaryFile() as tar_file:
            self.transmission_progress.update(
                download_task_id,
                description=f'Downloading model "{_tag}"',
                total=total_size_in_bytes,
                visible=True,
            )
            self.transmission_progress.start_task(download_task_id)
            for data in response.iter_content(block_size):
                self.transmission_progress.update(download_task_id, advance=len(data))
                tar_file.write(data)
            self.log_progress.add_task(
                f'[bold green]Finished downloading model "{_tag}" files'
            )
            tar_file.seek(0, 0)
            tar = tarfile.open(fileobj=tar_file, mode="r:gz")
            with self.spin(text=f'Extracting model "{_tag}" tar file'):
                with fs.open_fs("temp://") as temp_fs:
                    for member in tar.getmembers():
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        p = Path(member.name)
                        if p.parent != Path("."):
                            temp_fs.makedirs(str(p.parent), recreate=True)
                        temp_fs.writebytes(member.name, f.read())
                    model = Model.from_fs(temp_fs).save(model_store)
                    self.log_progress.add_task(
                        f'[bold green]Successfully pulled model "{_tag}"'
                    )
                    return model