from __future__ import annotations

import math
import os
import tarfile
import typing as t
import warnings
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from tempfile import mkstemp
from threading import Lock

import attrs
import fs
import httpx
from simple_di import Provide
from simple_di import inject

from ...exceptions import BentoMLException
from ...exceptions import NotFound
from ..configuration.containers import BentoMLContainer
from ..models import Model as StoredModel
from ..models import ModelStore
from ..tag import Tag
from ..utils.filesystem import safe_extract_tarfile
from .base import FILE_CHUNK_SIZE
from .base import UPLOAD_RETRY_COUNT
from .base import CallbackIOWrapper
from .base import Spinner
from .schemas.modelschemas import UploadStatus
from .schemas.schemasv1 import CompleteMultipartUploadSchema
from .schemas.schemasv1 import CompletePartSchema
from .schemas.schemasv1 import CreateModelRepositorySchema
from .schemas.schemasv1 import FinishUploadSchema
from .schemas.schemasv1 import ModelSchema
from .schemas.schemasv1 import PreSignMultipartUploadUrlSchema
from .schemas.schemasv1 import TransmissionStrategy

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from rich.progress import TaskID

    from _bentoml_sdk.models import Model

    from .client import RestApiClient
    from .schemas.schemasv1 import ModelWithRepositoryListSchema


@attrs.frozen
class ModelAPI:
    _client: RestApiClient = attrs.field(repr=False)
    spinner: Spinner = attrs.field(repr=False, factory=Spinner)
    _lock: Lock = attrs.field(repr=False, init=False, factory=Lock)

    def push(
        self,
        model: Model[t.Any],
        *,
        force: bool = False,
        threads: int = 10,
    ) -> None:
        """Push a model to remote model store

        Args:
            model: The model to push
            force: Whether to force push the model
            threads: The number of threads to use for the push
        """
        with self.spinner:
            upload_task_id = self.spinner.transmission_progress.add_task(
                f'Pushing model "{model}"', start=False, visible=False
            )
            self._do_push_model(model, upload_task_id, force=force, threads=threads)

    @inject
    def _do_push_model(
        self,
        model: Model[t.Any],
        upload_task_id: TaskID,
        *,
        force: bool = False,
        threads: int = 10,
        bentoml_tmp_dir: str = Provide[BentoMLContainer.tmp_bento_store_dir],
    ):
        from _bentoml_sdk.models import BentoModel

        rest_client = self._client

        model_info = model.to_info()
        name = model_info.tag.name
        version = model_info.tag.version
        if version is None:
            raise BentoMLException(f'Model "{model}" version cannot be None')

        with self._lock:
            # Models might be pushed by multiple threads at the same time
            # when they are under the same model repository, race condition
            # might happen when creating the model repository. So we need to
            # protect it with a lock.
            with self.spinner.spin(text=f'Fetching model repository "{name}"'):
                model_repository = rest_client.v1.get_model_repository(
                    model_repository_name=name
                )
            if not model_repository:
                with self.spinner.spin(
                    text=f'Model repository "{name}" not found, creating now..'
                ):
                    model_repository = rest_client.v1.create_model_repository(
                        req=CreateModelRepositorySchema(name=name, description="")
                    )
        with self.spinner.spin(
            text=f'Try fetching model "{model}" from remote model store..'
        ):
            remote_model = rest_client.v1.get_model(
                model_repository_name=name, version=version
            )
        if (
            not force
            and remote_model
            and remote_model.upload_status == UploadStatus.SUCCESS.value
        ):
            self.spinner.log(
                f'[bold blue]Model "{model}" already exists in remote model store, skipping'
            )
            return
        if not remote_model:
            with self.spinner.spin(
                text=f'Registering model "{model}" with remote model store..'
            ):
                remote_model = rest_client.v1.create_model(
                    model_repository_name=model_repository.name,
                    req=model.to_create_schema(),
                )
        if not isinstance(model, BentoModel):
            self.spinner.log(f"[bold blue]Skip uploading non-bentoml model {model}")
            return
        assert model.stored is not None
        transmission_strategy: TransmissionStrategy = "proxy"
        presigned_upload_url: str | None = None

        if remote_model.transmission_strategy is not None:
            transmission_strategy = remote_model.transmission_strategy
        else:
            with self.spinner.spin(
                text=f'Getting a presigned upload url for Model "{model.tag}" ..'
            ):
                remote_model = rest_client.v1.presign_model_upload_url(
                    model_repository_name=model_repository.name, version=version
                )
                if remote_model.presigned_upload_url:
                    transmission_strategy = "presigned_url"
                    presigned_upload_url = remote_model.presigned_upload_url

        def io_cb(x: int):
            self.spinner.transmission_progress.update(upload_task_id, advance=x)

        fd, tar_name = mkstemp(
            prefix="bentoml-model-", suffix=".tar", dir=bentoml_tmp_dir
        )
        tar_io = os.fdopen(fd, "wb+")
        try:
            with self.spinner.spin(
                text=f'Creating tar archive for model "{model.tag}"..'
            ):
                with tarfile.open(fileobj=tar_io, mode="w:") as tar:
                    tar.add(model.stored.path, arcname="./")
            with self.spinner.spin(text=f'Start uploading model "{model.tag}"..'):
                rest_client.v1.start_upload_model(
                    model_repository_name=model_repository.name, version=version
                )
            file_size = tar_io.tell()
            self.spinner.transmission_progress.update(
                upload_task_id,
                description=f'Uploading model "{model.tag}"',
                total=file_size,
                visible=True,
            )
            self.spinner.transmission_progress.start_task(upload_task_id)
            io_with_cb = CallbackIOWrapper(tar_io, read_cb=io_cb)

            if transmission_strategy == "proxy":
                try:
                    rest_client.v1.upload_model(
                        model_repository_name=model_repository.name,
                        version=version,
                        data=io_with_cb,
                    )
                except Exception as e:  # pylint: disable=broad-except
                    self.spinner.log(
                        f'[bold red]:police_car_light: Failed to upload model "{model.tag}"'
                    )
                    raise e
                self.spinner.log(
                    f'[bold green]:white_check_mark: Successfully pushed model "{model.tag}"'
                )
                return
            finish_req = FinishUploadSchema(
                status=UploadStatus.SUCCESS.value, reason=""
            )
            try:
                if presigned_upload_url is not None:
                    resp = httpx.put(
                        presigned_upload_url, content=io_with_cb, timeout=36000
                    )
                    if resp.status_code != 200:
                        finish_req = FinishUploadSchema(
                            status=UploadStatus.FAILED.value,
                            reason=resp.text,
                        )
                else:
                    with self.spinner.spin(
                        text=f'Start multipart uploading Model "{model.tag}"...'
                    ):
                        remote_model = rest_client.v1.start_model_multipart_upload(
                            model_repository_name=model_repository.name,
                            version=version,
                        )
                        if not remote_model.upload_id:
                            raise BentoMLException(
                                f'Failed to start multipart upload for model "{model.tag}", upload_id is empty'
                            )

                        upload_id: str = remote_model.upload_id

                    chunks_count = math.ceil(file_size / FILE_CHUNK_SIZE)
                    tar_io.close()

                    def chunk_upload(
                        upload_id: str, chunk_number: int
                    ) -> FinishUploadSchema | tuple[str, int]:
                        with self.spinner.spin(
                            text=f'({chunk_number}/{chunks_count}) Presign multipart upload url of model "{model.tag}"...'
                        ):
                            remote_model = (
                                rest_client.v1.presign_model_multipart_upload_url(
                                    model_repository_name=model_repository.name,
                                    version=version,
                                    req=PreSignMultipartUploadUrlSchema(
                                        upload_id=upload_id,
                                        part_number=chunk_number,
                                    ),
                                )
                            )

                        with self.spinner.spin(
                            text=f'({chunk_number}/{chunks_count}) Uploading chunk of model "{model.tag}"...'
                        ):
                            with open(tar_name, "rb") as f:
                                chunk_io = CallbackIOWrapper(
                                    f,
                                    read_cb=io_cb,
                                    start=(chunk_number - 1) * FILE_CHUNK_SIZE,
                                    end=chunk_number * FILE_CHUNK_SIZE
                                    if chunk_number < chunks_count
                                    else None,
                                )

                                for i in range(UPLOAD_RETRY_COUNT):
                                    resp = httpx.put(
                                        remote_model.presigned_upload_url,
                                        content=chunk_io,
                                        timeout=36000,
                                    )
                                    if resp.status_code == 200:
                                        break
                                    if i == UPLOAD_RETRY_COUNT - 1:
                                        return FinishUploadSchema(
                                            status=UploadStatus.FAILED.value,
                                            reason=resp.text,
                                        )
                                    else:  # retry and reset and update progress
                                        read = chunk_io.reset()
                                        self.spinner.transmission_progress.update(
                                            upload_task_id, advance=-read
                                        )
                                return resp.headers["ETag"], chunk_number

                    futures_: list[Future[FinishUploadSchema | tuple[str, int]]] = []

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
                        if isinstance(result, FinishUploadSchema):
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

                    with self.spinner.spin(
                        text=f'Completing multipart upload of model "{model.tag}"...'
                    ):
                        remote_model = rest_client.v1.complete_model_multipart_upload(
                            model_repository_name=model_repository.name,
                            version=version,
                            req=CompleteMultipartUploadSchema(
                                upload_id=upload_id,
                                parts=parts,
                            ),
                        )

            except Exception as e:  # pylint: disable=broad-except
                finish_req = FinishUploadSchema(
                    status=UploadStatus.FAILED.value,
                    reason=str(e),
                )
            if finish_req.status == UploadStatus.FAILED.value:
                self.spinner.log(f'[bold red]Failed to upload model "{model.tag}"')
            with self.spinner.spin(
                text="Submitting upload status to remote model store"
            ):
                rest_client.v1.finish_upload_model(
                    model_repository_name=model_repository.name,
                    version=version,
                    req=finish_req,
                )

            if finish_req.status != UploadStatus.SUCCESS.value:
                self.spinner.log(
                    f'[bold red]Failed pushing model "{model.tag}" : {finish_req.reason}'
                )
                raise BentoMLException(f'Failed to upload model "{model.tag}"')
            else:
                self.spinner.log(f'[bold green]Successfully pushed model "{model.tag}"')
        finally:
            try:
                tar_io.close()
            except OSError:
                pass
            os.unlink(tar_name)

    @inject
    def pull(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
        model_store: ModelStore = Provide[BentoMLContainer.model_store],
        query: str | None = None,
    ) -> StoredModel | None:
        """Pull a model from remote model store

        Args:
            tag: The tag of the model to pull
            force: Whether to force pull the model
            model_store: The model store to pull the model to
            query: The query to use for the pull

        Returns:
            The pulled model
        """
        with self.spinner:
            download_task_id = self.spinner.transmission_progress.add_task(
                f'Pulling model "{tag}"', start=False, visible=False
            )
            return self._do_pull_model(
                tag,
                download_task_id,
                force=force,
                model_store=model_store,
                query=query,
            )

    @inject
    def _do_pull_model(
        self,
        tag: str | Tag,
        download_task_id: TaskID,
        *,
        force: bool = False,
        model_store: ModelStore = Provide[BentoMLContainer.model_store],
        query: str | None = None,
    ) -> StoredModel | None:
        rest_client = self._client
        _tag = Tag.from_taglike(tag)
        try:
            model = model_store.get(_tag)
        except NotFound:
            model = None
        else:
            if _tag.version not in (None, "latest"):
                if not force:
                    self.spinner.log(
                        f'[bold blue]Model "{tag}" already exists locally, skipping'
                    )
                    return model
                else:
                    model_store.delete(tag)
        name = _tag.name
        version = _tag.version
        if version in (None, "latest"):
            latest_model = rest_client.v1.get_latest_model(name, query=query)
            if latest_model is None:
                raise BentoMLException(
                    f'Model "{_tag}" not found on remote model store, you may need to specify a version'
                )
            if model is not None:
                if not force and latest_model.build_at < model.creation_time:
                    self.spinner.log(
                        f'[bold blue]Newer version of model "{name}" exists locally, skipping'
                    )
                    return model
                if model.tag.version == latest_model.version:
                    if not force:
                        self.spinner.log(
                            f'[bold blue]Model "{model.tag}" already exists locally, skipping'
                        )
                        return model
                    else:
                        model_store.delete(model.tag)
            version = latest_model.version
        elif query:
            warnings.warn(
                "`query` is ignored when model version is specified", UserWarning
            )

        with self.spinner.spin(
            text=f'Getting a presigned download url for model "{_tag}"..'
        ):
            remote_model = rest_client.v1.presign_model_download_url(name, version)

        if not remote_model:
            raise BentoMLException(f'Model "{_tag}" not found on remote model store')
        if remote_model.manifest.metadata.get("registry") == "huggingface":
            self.spinner.log(f"[bold blue]No content to download for model {_tag}")
            return
        # Download model files from remote model store
        transmission_strategy: TransmissionStrategy = "proxy"
        presigned_download_url: str | None = None

        if remote_model.transmission_strategy is not None:
            transmission_strategy = remote_model.transmission_strategy
        else:
            with self.spinner.spin(
                text=f'Getting a presigned download url for model "{_tag}"'
            ):
                remote_model = rest_client.v1.presign_model_download_url(name, version)
                if remote_model.presigned_download_url:
                    presigned_download_url = remote_model.presigned_download_url
                    transmission_strategy = "presigned_url"

        if transmission_strategy == "proxy":
            response_ctx = rest_client.v1.download_model(
                model_repository_name=name, version=version
            )
        else:
            if presigned_download_url is None:
                with self.spinner.spin(
                    text=f'Getting a presigned download url for model "{_tag}"'
                ):
                    remote_model = rest_client.v1.presign_model_download_url(
                        name, version
                    )
                    presigned_download_url = remote_model.presigned_download_url

            response_ctx = httpx.stream("GET", presigned_download_url)

        with NamedTemporaryFile() as tar_file:
            with response_ctx as response:
                if response.status_code != 200:
                    response.read()
                    raise BentoMLException(
                        f'Failed to download model "{_tag}": {response.text}'
                    )

                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                self.spinner.transmission_progress.update(
                    download_task_id,
                    description=f'Downloading model "{_tag}"',
                    total=total_size_in_bytes,
                    visible=True,
                )
                self.spinner.transmission_progress.start_task(download_task_id)
                for data in response.iter_bytes(block_size):
                    self.spinner.transmission_progress.update(
                        download_task_id, advance=len(data)
                    )
                    tar_file.write(data)

            self.spinner.log(f'[bold green]Finished downloading model "{_tag}" files')
            tar_file.seek(0, 0)
            tar = tarfile.open(fileobj=tar_file, mode="r")
            with self.spinner.spin(text=f'Extracting model "{_tag}" tar file'):
                with fs.open_fs("temp://") as temp_fs:
                    safe_extract_tarfile(tar, temp_fs.getsyspath("/"))
                    model = StoredModel.from_fs(temp_fs).save(model_store)
                    self.spinner.log(f'[bold green]Successfully pulled model "{_tag}"')
                    return model

    def list(self) -> ModelWithRepositoryListSchema:
        """List all models in the remote model store

        Returns:
            The list of models
        """
        res = self._client.v1.get_models_list()
        if res is None:
            raise BentoMLException("List models request failed")

        res.items = [
            model
            for model in sorted(res.items, key=lambda x: x.created_at, reverse=True)
        ]
        return res

    def get(self, name: str, version: str | None = None) -> ModelSchema:
        """Get a model from the remote model store

        Args:
            tag: The tag of the model to get

        Returns:
            The model
        """
        if version is None or version == "latest":
            res = self._client.v1.get_latest_model(name)
        else:
            res = self._client.v1.get_model(name, version)
        if res is None:
            raise NotFound(f'Model "{name}:{version}" not found')
        return res
