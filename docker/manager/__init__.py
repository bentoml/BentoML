import logging

import click
import simple_di
from rich.logging import RichHandler
from rich.traceback import install

__version__: str = "1.1.0"

install(suppress=[click, simple_di])

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
