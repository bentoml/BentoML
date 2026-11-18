from __future__ import annotations

import io

import yaml

from bentoml._internal.configuration import load_config
from bentoml._internal.configuration import save_config
from bentoml._internal.configuration.containers import BentoMLContainer


def test_save_config_writes_resolved_config_dict():
    buf = io.StringIO()
    save_config(buf)
    assert yaml.safe_load(buf.getvalue()) == BentoMLContainer.config.get()


def test_save_config_round_trips(tmp_path):
    before = BentoMLContainer.config.get()
    path = tmp_path / "bentoml.yaml"
    with path.open("w") as f:
        save_config(f)
    load_config(str(path))
    assert BentoMLContainer.config.get() == before
