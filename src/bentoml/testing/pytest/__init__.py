from __future__ import annotations

import os
import tempfile

from ..utils import run_in_bazel
from ..._internal.models import ModelContext

__all__ = ["TEST_MODEL_CONTEXT", "set_huggingface_envar"]


TEST_MODEL_CONTEXT = ModelContext(
    framework_name="testing",
    framework_versions={"testing": "v1"},
)


def set_huggingface_envar():
    # We will create temp folder for caching, as we don't really care about caching
    # for testing.
    hf_home = tempfile.mkdtemp("huggingface")
    if run_in_bazel():
        # We need to set the cache directory for transformers here at test startup.
        hf_home = os.path.join(os.getcwd(), ".cache", "huggingface")
    hub_cache = os.path.join(hf_home, "hub")
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache

    [os.makedirs(d, exist_ok=True) for d in [hf_home, hub_cache]]
