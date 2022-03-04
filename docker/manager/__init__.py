import logging

from dotenv import load_dotenv
from rich.logging import RichHandler
from manager._container import ManagerContainer

__version__: str = "0.1.0"


load_dotenv(dotenv_path=ManagerContainer.docker_dir.joinpath(".env").as_posix())

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# TODO: 3.10 fails on conda authorization, on i386
# Could due to miniconda doesn't have build for i386 yet
SUPPORTED_PYTHON_VERSION = ["3.7", "3.8", "3.9", "3.10"]
SUPPORTED_OS_RELEASES = [
    "debian11",
    "debian10",
    "ubi8",
    "ubi7",
    "amazonlinux2",
    "alpine3.14",
]
DOCKERFILE_BUILD_HIERARCHY = ("base", "runtime", "cudnn", "devel")
