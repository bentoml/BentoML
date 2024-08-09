from __future__ import annotations

import attrs

from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.tag import Tag

from .base import Model

CONFIG_FILE = "config.json"


@attrs.frozen
class HuggingFaceModel(Model[str]):
    repo_id: str
    rev: str = "main"

    @property
    def revision(self) -> str:
        from huggingface_hub import get_hf_file_metadata
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(self.repo_id, CONFIG_FILE, revision=self.rev)
        metadata = get_hf_file_metadata(url)
        return metadata.commit_hash

    def resolve(self) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(self.repo_id, revision=self.rev)

    def to_info(self, alias: str | None = None) -> BentoModelInfo:
        tag = Tag(self.repo_id, self.revision)
        return BentoModelInfo(tag, registry="huggingface", alias=alias)
