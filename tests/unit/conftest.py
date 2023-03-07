# pylint: disable=unused-argument
from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import yaml
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.fixtures import FixtureRequest


@pytest.fixture(scope="function")
def reload_directory(
    request: FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> t.Generator[Path, None, None]:
    """
    This fixture will create an example bentoml working file directory
    and yield the results directory
    ./
    ├── models/  # mock default bentoml home models directory
    ├── [fdir, fdir_one, fdir_two]/
    │   ├── README.md
        ├── subdir/
        │   ├── README.md
    │   │   └── app.py
    │   ├── somerust.rs
    │   └── app.py
    ├── README.md
    ├── .bentoignore
    ├── bentofile.yaml
    ├── fname.ipynb
    ├── requirements.txt
    ├── service.py
    └── train.py
    """
    from bentoml._internal.utils import bentoml_cattr
    from bentoml._internal.bento.build_config import BentoBuildConfig

    root = tmp_path_factory.mktemp("reload_directory")
    # create a models directory
    root.joinpath("models").mkdir()

    # enable this fixture to use with unittest.TestCase
    if request.cls is not None:
        request.cls.reload_directory = root

    root_file = [
        "README.md",
        "requirements.txt",
        "service.py",
        "train.py",
        "fname.ipynb",
    ]

    for f in root_file:
        p = root.joinpath(f)
        p.touch()
    build_config = BentoBuildConfig(
        service="service.py:svc",
        description="A mock service",
        exclude=["*.rs"],
    ).with_defaults()
    bentofile = root / "bentofile.yaml"
    bentofile.touch()
    with bentofile.open("w", encoding="utf-8") as f:
        yaml.safe_dump(bentoml_cattr.unstructure(build_config), f)

    custom_library = ["fdir", "fdir_one", "fdir_two"]
    for app in custom_library:
        ap = root.joinpath(app)
        ap.mkdir()
        dir_files: list[tuple[str, list[t.Any]]] = [
            ("README.md", []),
            ("subdir", ["README.md", "app.py"]),
            ("lib.rs", []),
            ("app.py", []),
        ]
        for name, maybe_files in dir_files:
            if maybe_files:
                dpath = ap.joinpath(name)
                dpath.mkdir()
                for f in maybe_files:
                    p = dpath.joinpath(f)
                    p.touch()
            else:
                p = ap.joinpath(name)
                p.touch()

    yield root


@pytest.fixture(scope="function", name="propagate_logs")
def fixture_propagate_logs() -> t.Generator[None, None, None]:
    """BentoML sets propagate to False by default, hence this fixture enable log propagation."""
    logger = logging.getLogger("bentoml")
    logger.propagate = True
    yield
    # restore propagate to False after tests
    logger.propagate = False
