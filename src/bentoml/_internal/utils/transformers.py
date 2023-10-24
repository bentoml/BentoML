from __future__ import annotations

import re
import typing as t
from pathlib import Path

from ...exceptions import MissingDependencyException


def extract_commit_hash(
    resolved_dir: str, regex_commit_hash: t.Pattern[str]
) -> str | None:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    modified from https://github.com/huggingface/transformers/blob/0b7b4429c78de68acaf72224eb6dae43616d820c/src/transformers/utils/hub.py#L219
    """

    resolved_dir = str(Path(resolved_dir).as_posix()) + "/"
    search = re.search(r"snapshots/([^/]+)/", resolved_dir)

    if search is None:
        return None

    commit_hash = search.groups()[0]
    return commit_hash if regex_commit_hash.match(commit_hash) else None


def is_huggingface_hub_available():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:  # pragma: no cover
        raise MissingDependencyException(
            "'huggingface_hub' is required in order to download pretrained models, install with 'pip install huggingface-hub'. For more information, refer to https://huggingface.co/docs/huggingface_hub/quick-start",
        )


def is_accelerate_available():
    try:
        import accelerate  # noqa: F401
    except ImportError:  # pragma: no cover
        raise MissingDependencyException(
            "'accelerate' is required in order to download pretrained models, install with 'pip install accelerate'.",
        )
