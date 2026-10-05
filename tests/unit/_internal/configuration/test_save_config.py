from __future__ import annotations

import io

import yaml

from bentoml._internal.configuration import save_config
from bentoml._internal.configuration.containers import BentoMLContainer


def test_save_config_writes_resolved_config_dict():
    buf = io.StringIO()
    save_config(buf)
    assert yaml.safe_load(buf.getvalue()) == BentoMLContainer.config.get()
