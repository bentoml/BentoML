
from typing import Optional, Union

import pytest
from _pytest.config import Config, ExitCode
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureDef, SubRequest

def pytest_addoption(parser: Parser) -> None:
    ...

@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: FixtureDef[object], request: SubRequest) -> Optional[object]:
    ...

@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: Config) -> Optional[Union[int, ExitCode]]:
    ...

