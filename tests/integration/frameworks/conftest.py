from __future__ import annotations

import os
import pkgutil
from types import ModuleType
from importlib import import_module

import pytest

from .models import FrameworkTestModel


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--framework", action="store", default=None)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    framework_name: str = metafunc.config.getoption("framework")  # type: ignore

    if "framework" in metafunc.fixturenames and "test_model" in metafunc.fixturenames:
        metafunc.parametrize("framework,test_model", test_inputs(framework_name))
    elif "framework" in metafunc.fixturenames:
        metafunc.parametrize(
            "framework", [inp[0] for inp in test_inputs(framework_name)]
        )


def test_inputs(framework: str | None) -> list[tuple[ModuleType, FrameworkTestModel]]:
    if framework is None:
        frameworks = [
            name
            for _, name, _ in pkgutil.iter_modules(
                [os.path.join(os.path.dirname(__file__), "models")]
            )
        ]
    else:
        frameworks = [framework]

    input_modules: list[ModuleType] = []
    for framework_name in frameworks:
        try:
            input_modules.append(
                import_module(
                    f".{framework_name}", "tests.integration.frameworks.models"
                )
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Failed to find test module for framework {framework_name} (tests.integration.frameworks.models.{framework_name})"
            ) from e

    return [
        (module.framework, _model)
        for module in input_modules
        for _model in module.models
    ]
