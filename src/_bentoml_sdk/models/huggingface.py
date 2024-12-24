from __future__ import annotations

import hashlib
import json
import os
import shutil
import typing as t
from functools import cached_property

import attrs
from fs.base import FS

from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.cloud.schemas.modelschemas import ModelManifestSchema
from bentoml._internal.cloud.schemas.schemasv1 import CreateModelSchema
from bentoml._internal.models.model import ModelContext
from bentoml._internal.tag import Tag
from bentoml._internal.types import PathType

from .base import Model

if t.TYPE_CHECKING:
    from huggingface_hub import HfApi

CONFIG_FILE = "config.json"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"


@attrs.define(unsafe_hash=True)
class HuggingFaceModel(Model[str]):
    """A model reference to a Hugging Face model.

    Args:
        model_id (str): The model tag. E.g. "google-bert/bert-base-uncased".
        revision (str, optional): The revision to use. Defaults to "main".
        endpoint (str, optional): The Hugging Face endpoint to use. Defaults to https://huggingface.co.
        include (List[str], optional): The files to include. Defaults to all files.
        exclude (List[str], optional): The files to exclude. Defaults to no files.

    Returns:
        str: The downloaded model path.
    """

    model_id: str
    revision: str = "main"
    endpoint: t.Optional[str] = attrs.field(factory=lambda: os.getenv("HF_ENDPOINT"))
    include: t.Optional[t.List[str]] = None
    exclude: t.Optional[t.List[str]] = None

    @cached_property
    def _hf_api(self) -> HfApi:
        from huggingface_hub import HfApi

        return HfApi(endpoint=self.endpoint)

    @cached_property
    def commit_hash(self) -> str:
        return (
            self._hf_api.model_info(self.model_id, revision=self.revision).sha
            or self.revision
        )

    def resolve(self, base_path: t.Union[PathType, FS, None] = None) -> str:
        from huggingface_hub import snapshot_download

        if isinstance(base_path, FS):
            base_path = base_path.getsyspath("/")

        snapshot_path = snapshot_download(
            self.model_id,
            revision=self.revision,
            endpoint=self.endpoint,
            cache_dir=os.getenv("BENTOML_HF_CACHE_DIR"),
            allow_patterns=self.include,
            ignore_patterns=self.exclude,
        )
        if base_path is not None:
            model_path = os.path.dirname(os.path.dirname(snapshot_path))
            os.makedirs(base_path, exist_ok=True)
            shutil.copytree(
                model_path,
                os.path.join(base_path, os.path.basename(model_path)),
                symlinks=True,
                dirs_exist_ok=True,
            )
        return snapshot_path

    def to_info(self, alias: str | None = None) -> BentoModelInfo:
        model_id = self.model_id.lower()
        metadata = {
            "model_id": model_id,
            "revision": self.commit_hash,
            "endpoint": self.endpoint or DEFAULT_HF_ENDPOINT,
            "include": self.include,
            "exclude": self.exclude,
        }
        content_hash = hashlib.md5(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        tag = Tag(model_id.replace("/", "--"), content_hash)
        return BentoModelInfo(
            tag, alias=alias, registry="huggingface", metadata=metadata
        )

    @classmethod
    def from_info(cls, info: BentoModelInfo) -> HuggingFaceModel:
        if not info.metadata:
            name = info.tag.name
            return cls(model_id=name.replace("--", "/"))
        model = cls(
            model_id=info.metadata["model_id"],
            revision=info.metadata["revision"],
            endpoint=info.metadata["endpoint"],
            include=info.metadata.get("include"),
            exclude=info.metadata.get("exclude"),
        )
        # the commit hash is frozen in the model info, update the cache directly
        model.__dict__.update(commit_hash=info.metadata["revision"])
        return model

    def _get_model_size(self, revision: str) -> int:
        from huggingface_hub.utils import filter_repo_objects

        info = self._hf_api.model_info(
            self.model_id, revision=revision, files_metadata=True
        )
        filtered_files = filter_repo_objects(
            items=info.siblings or [],
            allow_patterns=self.include,
            ignore_patterns=self.exclude,
            key=lambda f: f.rfilename,
        )
        return sum((file.size or 0) for file in filtered_files)

    def to_create_schema(self) -> CreateModelSchema:
        context = ModelContext(framework_name="huggingface", framework_versions={})
        endpoint = self.endpoint or DEFAULT_HF_ENDPOINT
        revision = self.commit_hash
        url = f"{endpoint}/{self.model_id}/tree/{revision}"
        metadata = {
            "registry": "huggingface",
            "model_id": self.model_id,
            "revision": revision,
            "endpoint": endpoint,
            "include": self.include,
            "exclude": self.exclude,
            "url": url,
        }
        return CreateModelSchema(
            description="",
            version=self.to_info().tag.version or revision,
            manifest=ModelManifestSchema(
                module="",
                metadata=metadata,
                api_version="v1",
                bentoml_version=context.bentoml_version,
                size_bytes=self._get_model_size(revision),
                context=context.to_dict(),
                options={},
            ),
        )

    def __str__(self) -> str:
        return f"{self.model_id}:{self.revision}"
