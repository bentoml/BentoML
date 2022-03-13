import logging

import click
from rich.logging import RichHandler
from rich.traceback import install

__version__: str = "1.1.0"

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# add traceback when in interactive shell for development
install(suppress=[click])
