from __future__ import annotations

import math
import tarfile
import typing as t
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import attrs
import fs
import httpx
from simple_di import Provide
from simple_di import inject

from ...exceptions import BentoMLException
from ...exceptions import NotFound
from ..bento import Bento
from ..bento import BentoStore
from ..configuration.containers import BentoMLContainer
from ..tag import Tag
from .base import FILE_CHUNK_SIZE
from .base import UPLOAD_RETRY_COUNT
from .base import CallbackIOWrapper
from .base import Spinner
from .model import ModelAPI
from .schemas.modelschemas import BentoApiSchema
from .schemas.modelschemas import BentoRunnerResourceSchema
from .schemas.modelschemas import BentoRunnerSchema
from .schemas.modelschemas import BentoUploadStatus
from .schemas.schemasv1 import BentoManifestSchema
from .schemas.schemasv1 import BentoSchema
from .schemas.schemasv1 import CompleteMultipartUploadSchema
from .schemas.schemasv1 import CompletePartSchema
from .schemas.schemasv1 import CreateBentoRepositorySchema
from .schemas.schemasv1 import CreateBentoSchema
from .schemas.schemasv1 import FinishUploadBentoSchema
from .schemas.schemasv1 import LabelItemSchema
from .schemas.schemasv1 import PreSignMultipartUploadUrlSchema
from .schemas.schemasv1 import TransmissionStrategy
from .schemas.schemasv1 import UpdateBentoSchema

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from rich.progress import TaskID

    from _bentoml_sdk.models import Model

    from .client import RestApiClient
    from .schemas.schemasv1 import BentoWithRepositoryListSchema


@attrs.frozen
class BentoAPI:
    _client: RestApiClient = attrs.field(repr=False)
    spinner: Spinner = attrs.field(repr=False, factory=Spinner)

    @inject
    def push(
        self,
        bento: Bento,
        *,
        force: bool = False,
        bare: bool = False,
        threads: int = 10,
    ) -> None:
        """Push a Bento to the remote Bento store

        Args:
            bento: The Bento to push
            force: Whether to force push the Bento
            bare: If true, only push the Bento manifest
            threads: The number of threads to use for the push
        """
        with self.spinner:
            upload_task_id = self.spinner.transmission_progress.add_task(
                f'Pushing Bento "{bento.tag}"', start=False, visible=False
            )
            self._do_push_bento(
                bento, upload_task_id, force=force, threads=threads, bare=bare
            )
            self.spinner.log(f'✅ Pushed Bento "{bento.tag}"')

    @inject
    def _do_push_bento(
        self,
        bento: Bento,
        upload_task_id: TaskID,
        *,
        force: bool = False,
        threads: int = 10,
        bare: bool = False,
        bentoml_tmp_dir: str = Provide[BentoMLContainer.tmp_bento_store_dir],
    ):
        rest_client = self._client
        from _bentoml_sdk.models import BentoModel
        from _bentoml_sdk.models import HuggingFaceModel

        name = bento.tag.name
        version = bento.tag.version
        if version is None:
            raise BentoMLException(f"Bento {bento.tag} version cannot be None")
        info = bento.info
        models_to_push: list[Model[t.Any]] = []
        for model in info.all_models:
            if model.registry == "huggingface":
                models_to_push.append(HuggingFaceModel.from_info(model))
            else:
                model = BentoModel(model.tag)
                if model.stored is not None:
                    models_to_push.append(model)
        model_api = ModelAPI(self._client, self.spinner)

        def push_model(model: Model[t.Any]) -> None:
            model_upload_task_id = self.spinner.transmission_progress.add_task(
                f'Pushing model "{model}"', start=False, visible=False
            )
            model_api._do_push_model(
                model,
                model_upload_task_id,
                force=force,
                threads=threads,
            )

        with ThreadPoolExecutor(max_workers=max(len(models_to_push), 1)) as executor:
            list(executor.map(push_model, models_to_push))

        with self.spinner.spin(text=f'Fetching Bento repository "{name}"'):
            bento_repository = rest_client.v1.get_bento_repository(
                bento_repository_name=name
            )
        if not bento_repository:
            with self.spinner.spin(
                text=f'Bento repository "{name}" not found, creating now..'
            ):
                bento_repository = rest_client.v1.create_bento_repository(
                    req=CreateBentoRepositorySchema(name=name, description="")
                )
        with self.spinner.spin(
            text=f'Try fetching Bento "{bento.tag}" from remote Bento store..'
        ):
            remote_bento = rest_client.v1.get_bento(
                bento_repository_name=name, version=version
            )
        if (
            not force
            and remote_bento
            and remote_bento.upload_status == BentoUploadStatus.SUCCESS.value
        ):
            self.spinner.log(
                f'[bold blue]Push skipped: Bento "{bento.tag}" already exists in remote Bento store'
            )
            return
        labels: list[LabelItemSchema] = [
            LabelItemSchema(key=key, value=value) for key, value in info.labels.items()
        ]
        apis: dict[str, BentoApiSchema] = {}
        models = [str(m.tag) for m in info.all_models]
        runners = [
            BentoRunnerSchema(
                name=r.name,
                runnable_type=r.runnable_type,
                models=r.models,
                resource_config=(
                    BentoRunnerResourceSchema(
                        cpu=r.resource_config.get("cpu"),
                        nvidia_gpu=r.resource_config.get("nvidia.com/gpu"),
                        custom_resources=r.resource_config.get("custom_resources"),
                    )
                    if r.resource_config
                    else None
                ),
            )
            for r in info.runners
        ]
        manifest = BentoManifestSchema(
            name=info.name,
            entry_service=info.entry_service,
            service=info.service,
            bentoml_version=info.bentoml_version,
            apis=apis,
            models=models,
            runners=runners,
            size_bytes=bento.total_size(),
            services=info.services,
            envs=info.envs,
            schema=info.schema,
        )
        if not remote_bento:
            with self.spinner.spin(
                text=f'Registering Bento "{bento.tag}" with remote Bento store..'
            ):
                remote_bento = rest_client.v1.create_bento(
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
            with self.spinner.spin(text=f'Updating Bento "{bento.tag}"..'):
                remote_bento = rest_client.v1.update_bento(
                    bento_repository_name=bento_repository.name,
                    version=version,
                    req=UpdateBentoSchema(
                        manifest=manifest,
                        labels=labels,
                    ),
                )
        if bare:
            with self.spinner.spin(
                text="Submitting upload status to remote Bento store"
            ):
                rest_client.v1.finish_upload_bento(
                    bento_repository_name=bento_repository.name,
                    version=version,
                    req=FinishUploadBentoSchema(
                        status=BentoUploadStatus.SUCCESS.value,
                        reason="bento for development",
                    ),
                )
            return
        transmission_strategy: TransmissionStrategy = "proxy"
        presigned_upload_url: str | None = None

        if remote_bento.transmission_strategy is not None:
            transmission_strategy = remote_bento.transmission_strategy
        else:
            with self.spinner.spin(
                text=f'Getting a presigned upload url for bento "{bento.tag}" ..'
            ):
                remote_bento = rest_client.v1.presign_bento_upload_url(
                    bento_repository_name=bento_repository.name, version=version
                )
                if remote_bento.presigned_upload_url:
                    transmission_strategy = "presigned_url"
                    presigned_upload_url = remote_bento.presigned_upload_url

        def io_cb(x: int):
            self.spinner.transmission_progress.update(upload_task_id, advance=x)

        with NamedTemporaryFile(
            prefix="bentoml-bento-", suffix=".tar", dir=bentoml_tmp_dir
        ) as tar_io:
            with self.spinner.spin(
                text=f'Creating tar archive for bento "{bento.tag}"..'
            ):
                with tarfile.open(fileobj=tar_io, mode="w:") as tar:

                    def filter_(
                        tar_info: tarfile.TarInfo,
                    ) -> t.Optional[tarfile.TarInfo]:
                        if tar_info.path == "./models" or tar_info.path.startswith(
                            "./models/"
                        ):
                            return None
                        return tar_info

                    tar.add(bento.path, arcname="./", filter=filter_)

            with self.spinner.spin(text=f'Start uploading bento "{bento.tag}"..'):
                rest_client.v1.start_upload_bento(
                    bento_repository_name=bento_repository.name, version=version
                )
            file_size = tar_io.tell()
            io_with_cb = CallbackIOWrapper(tar_io, read_cb=io_cb)

            self.spinner.transmission_progress.update(
                upload_task_id, completed=0, total=file_size, visible=True
            )
            self.spinner.transmission_progress.start_task(upload_task_id)

            if transmission_strategy == "proxy":
                try:
                    rest_client.v1.upload_bento(
                        bento_repository_name=bento_repository.name,
                        version=version,
                        data=io_with_cb,
                    )
                except Exception as e:  # pylint: disable=broad-except
                    self.spinner.log(f'[bold red]Failed to upload bento "{bento.tag}"')
                    raise e
                self.spinner.log(f'[bold green]Successfully pushed bento "{bento.tag}"')
                return
            finish_req = FinishUploadBentoSchema(
                status=BentoUploadStatus.SUCCESS.value, reason=""
            )
            try:
                if presigned_upload_url is not None:
                    resp = httpx.put(
                        presigned_upload_url, content=io_with_cb, timeout=36000
                    )
                    if resp.status_code != 200:
                        finish_req = FinishUploadBentoSchema(
                            status=BentoUploadStatus.FAILED.value,
                            reason=resp.text,
                        )
                else:
                    with self.spinner.spin(
                        text=f'Start multipart uploading Bento "{bento.tag}"...'
                    ):
                        remote_bento = rest_client.v1.start_bento_multipart_upload(
                            bento_repository_name=bento_repository.name,
                            version=version,
                        )
                        if not remote_bento.upload_id:
                            raise BentoMLException(
                                f'Failed to start multipart upload for Bento "{bento.tag}", upload_id is empty'
                            )

                        upload_id: str = remote_bento.upload_id

                    chunks_count = math.ceil(file_size / FILE_CHUNK_SIZE)
                    tar_io.file.close()

                    def chunk_upload(
                        upload_id: str, chunk_number: int
                    ) -> FinishUploadBentoSchema | tuple[str, int]:
                        with self.spinner.spin(
                            text=f'({chunk_number}/{chunks_count}) Presign multipart upload url of Bento "{bento.tag}"...'
                        ):
                            remote_bento = (
                                rest_client.v1.presign_bento_multipart_upload_url(
                                    bento_repository_name=bento_repository.name,
                                    version=version,
                                    req=PreSignMultipartUploadUrlSchema(
                                        upload_id=upload_id,
                                        part_number=chunk_number,
                                    ),
                                )
                            )
                        with self.spinner.spin(
                            text=f'({chunk_number}/{chunks_count}) Uploading chunk of Bento "{bento.tag}"...'
                        ):
                            with open(tar_io.name, "rb") as f:
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
                                        remote_bento.presigned_upload_url,
                                        content=chunk_io,
                                        timeout=36000,
                                    )
                                    if resp.status_code == 200:
                                        break
                                    if i == UPLOAD_RETRY_COUNT - 1:
                                        return FinishUploadBentoSchema(
                                            status=BentoUploadStatus.FAILED.value,
                                            reason=resp.text,
                                        )
                                    else:  # retry and reset and update progress
                                        read = chunk_io.reset()
                                        self.spinner.transmission_progress.update(
                                            upload_task_id, advance=-read
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

                    with self.spinner.spin(
                        text=f'Completing multipart upload of Bento "{bento.tag}"...'
                    ):
                        remote_bento = rest_client.v1.complete_bento_multipart_upload(
                            bento_repository_name=bento_repository.name,
                            version=version,
                            req=CompleteMultipartUploadSchema(
                                upload_id=upload_id,
                                parts=parts,
                            ),
                        )

            except Exception as e:  # pylint: disable=broad-except
                finish_req = FinishUploadBentoSchema(
                    status=BentoUploadStatus.FAILED.value,
                    reason=str(e),
                )
            if finish_req.status == BentoUploadStatus.FAILED.value:
                self.spinner.log(f'[bold red]Failed to upload Bento "{bento.tag}"')
            with self.spinner.spin(
                text="Submitting upload status to remote Bento store"
            ):
                rest_client.v1.finish_upload_bento(
                    bento_repository_name=bento_repository.name,
                    version=version,
                    req=finish_req,
                )
            if finish_req.status != BentoUploadStatus.SUCCESS.value:
                self.spinner.log(
                    f'[bold red]Failed pushing Bento "{bento.tag}": {finish_req.reason}'
                )
                raise BentoMLException(f'Failed to upload Bento "{bento.tag}"')
            else:
                self.spinner.log(f'[bold green]Successfully pushed Bento "{bento.tag}"')

    @inject
    def pull(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
        bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    ) -> Bento:
        """Pull a bento from remote bento store

        Args:
            tag: The tag of the bento to pull
            force: Whether to force pull the bento
            bento_store: The bento store to pull the bento to

        Returns:
            The pulled bento
        """
        with self.spinner:
            download_task_id = self.spinner.transmission_progress.add_task(
                f'Pulling bento "{tag}"', start=False, visible=False
            )
            return self._do_pull_bento(
                tag,
                download_task_id,
                force=force,
                bento_store=bento_store,
            )

    def _do_pull_bento(
        self,
        tag: str | Tag,
        download_task_id: TaskID,
        *,
        force: bool = False,
        bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    ) -> Bento:
        rest_client = self._client
        try:
            bento = bento_store.get(tag)
            if not force:
                self.spinner.log(
                    f'[bold blue]Bento "{tag}" exists in local bento store'
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

        with self.spinner.spin(text=f'Fetching bento "{_tag}"'):
            remote_bento = rest_client.v1.get_bento(
                bento_repository_name=name, version=version
            )
        if not remote_bento:
            raise BentoMLException(f'Bento "{_tag}" not found on remote Bento store')

        # Download bento files from remote Bento store
        transmission_strategy: TransmissionStrategy = "proxy"
        presigned_download_url: str | None = None

        if remote_bento.transmission_strategy is not None:
            transmission_strategy = remote_bento.transmission_strategy
        else:
            with self.spinner.spin(
                text=f'Getting a presigned download url for bento "{_tag}"'
            ):
                remote_bento = rest_client.v1.presign_bento_download_url(name, version)
                if remote_bento.presigned_download_url:
                    presigned_download_url = remote_bento.presigned_download_url
                    transmission_strategy = "presigned_url"

        if transmission_strategy == "proxy":
            response_ctx = rest_client.v1.download_bento(
                bento_repository_name=name,
                version=version,
            )
        else:
            if presigned_download_url is None:
                with self.spinner.spin(
                    text=f'Getting a presigned download url for bento "{_tag}"'
                ):
                    remote_bento = rest_client.v1.presign_bento_download_url(
                        name, version
                    )
                    presigned_download_url = remote_bento.presigned_download_url

            response_ctx = httpx.stream("GET", presigned_download_url)

        with NamedTemporaryFile() as tar_file:
            with response_ctx as response:
                if response.status_code != 200:
                    raise BentoMLException(
                        f'Failed to download bento "{_tag}": {response.text}'
                    )
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                self.spinner.transmission_progress.update(
                    download_task_id,
                    completed=0,
                    total=total_size_in_bytes,
                    visible=True,
                )
                self.spinner.transmission_progress.start_task(download_task_id)
                for data in response.iter_bytes(block_size):
                    self.spinner.transmission_progress.update(
                        download_task_id, advance=len(data)
                    )
                    tar_file.write(data)

            self.spinner.log(
                f'[bold green]Finished downloading all bento "{_tag}" files'
            )
            tar_file.seek(0, 0)
            tar = tarfile.open(fileobj=tar_file, mode="r")
            with self.spinner.spin(text=f'Extracting bento "{_tag}" tar file'):
                with fs.open_fs("temp://") as temp_fs:
                    for member in tar.getmembers():
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        p = Path(member.name)
                        if p.parent != Path("."):
                            temp_fs.makedirs(p.parent.as_posix(), recreate=True)
                        temp_fs.writebytes(member.name, f.read())
                    bento = Bento.from_fs(temp_fs)
                    bento = bento.save(bento_store)
                    self.spinner.log(f'[bold green]Successfully pulled bento "{_tag}"')
                    return bento

    def list(self) -> BentoWithRepositoryListSchema:
        """List all bentos in the remote bento store

        Returns:
            The list of bentos
        """
        res = self._client.v1.get_bentos_list()
        if res is None:
            raise BentoMLException("List bentos request failed")

        res.items = [
            bento
            for bento in sorted(res.items, key=lambda x: x.created_at, reverse=True)
        ]
        return res

    def get(self, name: str, version: str | None) -> BentoSchema:
        """Get a bento by name and version

        Args:
            name: The name of the bento
            version: The version of the bento

        Returns:
            The bento
        """
        rest_client = self._client
        if version is None or version == "latest":
            res = rest_client.v1.list_bentos(bento_repository_name=name)
            if res is None:
                raise BentoMLException("List bento request failed")
            return res.items[0]
        res = rest_client.v1.get_bento(bento_repository_name=name, version=version)
        if res is None:
            raise NotFound(f'Bento "{name}:{version}" not found')
        return res
