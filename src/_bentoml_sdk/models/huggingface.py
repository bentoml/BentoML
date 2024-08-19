from __future__ import annotations

import os
import shutil
import typing as t

import attrs
from fs.base import FS

from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.cloud.schemas.modelschemas import ModelManifestSchema
from bentoml._internal.cloud.schemas.schemasv1 import CreateModelSchema
from bentoml._internal.models.model import ModelContext
from bentoml._internal.tag import Tag
from bentoml._internal.types import PathType

from .base import Model

CONFIG_FILE = "config.json"


@attrs.frozen
class HuggingFaceModel(Model[str]):
    """A model reference to a Hugging Face model.

    Args:
        model_id (Tag): The model tag. E.g. "google-bert/bert-base-uncased".
            You can specify a rev or commit hash by appending it to the model name separated by a colon:
                google-bert/bert-base-uncased:main
                google-bert/bert-base-uncased:86b5e0934494bd15c9632b12f734a8a67f723594
        endpoint (str, optional): The Hugging Face endpoint to use. Defaults to https://huggingface.co.

    Returns:
        str: The downloaded model path.
    """

    tag: Tag = attrs.field(converter=Tag.from_taglike, alias="model_id")
    endpoint: str | None = attrs.field(factory=lambda: os.getenv("HF_ENDPOINT"))

    @property
    def revision(self) -> str:
        from huggingface_hub import get_hf_file_metadata
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(
            self.tag.name,
            CONFIG_FILE,
            revision=self.tag.version,
            endpoint=self.endpoint,
        )
        metadata = get_hf_file_metadata(url)
        return metadata.commit_hash

    def resolve(self, base_path: t.Union[PathType, FS, None] = None) -> str:
        from huggingface_hub import snapshot_download

        if isinstance(base_path, FS):
            base_path = base_path.getsyspath("/")

        snapshot_path = snapshot_download(
            self.tag.name,
            revision=self.tag.version,
            endpoint=self.endpoint,
            cache_dir=os.getenv("BENTOML_HF_CACHE_DIR"),
        )
        if base_path is not None:
            model_path = os.path.dirname(os.path.dirname(snapshot_path))
            os.makedirs(base_path, exist_ok=True)
            shutil.copytree(
                model_path,
                os.path.join(base_path, os.path.basename(model_path)),
                symlinks=True,
            )
        return snapshot_path

    def to_info(self, alias: str | None = None) -> BentoModelInfo:
        tag = Tag(self.tag.name, self.revision)
        return BentoModelInfo(
            tag, registry="huggingface", alias=alias, endpoint=self.endpoint
        )

    def to_create_schema(self) -> CreateModelSchema:
        context = ModelContext(framework_name="huggingface", framework_versions={})
        metadata = {"registry": "huggingface"}
        if self.endpoint is not None:
            metadata["endpoint"] = self.endpoint
        return CreateModelSchema(
            description="",
            version=self.revision,
            manifest=ModelManifestSchema(
                module="",
                metadata=metadata,
                api_version="v1",
                bentoml_version="",
                size_bytes=0,
                context=context.to_dict(),
                options={},
            ),
        )
