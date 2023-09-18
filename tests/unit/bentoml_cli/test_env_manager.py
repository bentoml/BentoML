from __future__ import annotations

import pytest

from bentoml_cli.env_manager import remove_env_arg

testdata = [
    (
        "bentoml serve --env conda iris_classifier".split(),
        "bentoml serve iris_classifier".split(),
    ),
    (
        "bentoml serve --env=conda iris_classifier".split(),
        "bentoml serve iris_classifier".split(),
    ),
    (
        "bentoml serve --env=conda env_conda_bento".split(),
        "bentoml serve env_conda_bento".split(),
    ),
]


@pytest.mark.parametrize("cmd_args, clean_cmd_args", testdata)
def test_remove_env_arg(cmd_args: list[str], clean_cmd_args: list[str]):
    assert remove_env_arg(cmd_args) == clean_cmd_args
