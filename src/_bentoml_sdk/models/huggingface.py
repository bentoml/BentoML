from __future__ import annotations

import os

import attrs

from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.tag import Tag

from .base import Model

CONFIG_FILE = "config.json"


@attrs.frozen
class HuggingFaceModel(Model[str]):
    repo_id: str
    rev: str = "main"
    endpoint: str | None = attrs.field(factory=lambda: os.getenv("HF_ENDPOINT"))

    @property
    def revision(self) -> str:
        from huggingface_hub import get_hf_file_metadata
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(
            self.repo_id, CONFIG_FILE, revision=self.rev, endpoint=self.endpoint
        )
        metadata = get_hf_file_metadata(url)
        return metadata.commit_hash

    def resolve(self) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.repo_id, revision=self.rev, endpoint=self.endpoint
        )

    def to_info(self, alias: str | None = None) -> BentoModelInfo:
        tag = Tag(self.repo_id, self.revision)
        return BentoModelInfo(
            tag, registry="huggingface", alias=alias, endpoint=self.endpoint
        )
