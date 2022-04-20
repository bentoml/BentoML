import logging
import traceback

import click
import simple_di
from rich.logging import RichHandler
from rich.traceback import install
from pathlib import Path

__version__: str = "1.1.0"

root_docker = Path(__file__).parent.parent

install(suppress=[click, simple_di, traceback])

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

if not root_docker.joinpath("generated").exists():
    root_docker.joinpath("generated").mkdir(exist_ok=True)
