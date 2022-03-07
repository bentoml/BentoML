import io
import sys
import json
import atexit
import typing as t
import logging
import traceback
import subprocess
from typing import TYPE_CHECKING
from pathlib import Path as pathlib_path
from functools import wraps

import attrs
from glom import glom
from glom import Path as glom_path
from glom import Assign
from glom import PathAccessError
from glom import PathAssignError
from plumbum import local
from manager.exceptions import ManagerException
from manager.exceptions import ManagerInvalidShellCmd

if TYPE_CHECKING:
    from manager._types import P
    from manager._types import GenericDict
    from manager._types import GenericList

logger = logging.getLogger(__name__)

__all__ = (
    "sprint",
    "jprint",
    "get_data",
    "walk",
    "set_data",
    "shellcmd",
    "graceful_exit",
    "as_posix",
)

SUPPORTED_ARCHITECTURE_TYPE = [
    "amd64",
    "arm32v7",
    "arm64v8",
    "ppc64le",
    "i386",
    "s390x",
]


def _generate_manifest(name: str):
    ...


def serialize_to_json(d: t.Dict[str, t.Any]) -> "GenericDict":
    if not all(hasattr(v, "__attrs_attrs__") for v in d.values()):
        # we won't try to serialize non-attrs class
        return d
    return {k: attrs.asdict(v) for k, v in d.items()}


def as_posix(*args: t.Any) -> str:
    return pathlib_path(*args).as_posix()


def walk(
    path: pathlib_path, exclude: t.Optional[t.Tuple[str]] = None
) -> t.Generator[pathlib_path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            if exclude is not None and p.name in exclude:
                continue
            else:
                yield from walk(p)
                continue
        yield p.resolve()


def sprint(*args: t.Any, **kwargs: t.Any) -> None:
    # stream logs inside docker container to sys.stderr
    print(*args, file=sys.stderr, flush=True, **kwargs)


def jprint(*args: t.Any, indent: int = 2, **kwargs: t.Any) -> None:
    sprint(json.dumps(args[0], indent=indent), *args[1:], **kwargs)


def get_docker_platform_mapping() -> t.Dict[str, str]:
    # hard code these platform, seems easy to manage
    formatted = "linux/{fmt}"

    def process(arch: str) -> str:
        if "arm64" in arch:
            fmt = "arm64/v8"
        elif "arm32" in arch:
            fmt = "/".join(arch.split("32"))
        elif "i386" in arch:
            fmt = arch[1:]
        else:
            fmt = arch
        return formatted.format(fmt=fmt)

    return {k: process(k) for k in SUPPORTED_ARCHITECTURE_TYPE}


def get_data(
    obj: "t.Union[GenericDict, t.MutableMapping[str, t.Any]]", *path: str
) -> "GenericDict":
    """
    Get data from the object by dotted path.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        data = glom(obj, glom_path(*path))
    except PathAccessError:
        logger.exception(
            "Exception occurred in "
            "get_data, unable to retrieve "
            f"{'.'.join([*path])} from {repr(obj)}"
        )
        exit(1)
    else:
        return data


def set_data(
    obj: "GenericDict", value: "t.Union[GenericDict, GenericList, str]", *path: str
) -> None:
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(glom_path(*path), value))
    except PathAssignError as err:
        logger.error(
            "Exception occurred in "
            "update_data, unable to update "
            f"{glom_path(*path)} with {value}"
        )
        raise ManagerException("Error while setting data to object.") from err


@attrs.define
class Output:
    returncode: int
    stderr: str
    stdout: str


def shellcmd(
    bin: str, *args: str, log_output: bool = True, shell: bool = False
) -> Output:
    stdout, stderr = "", ""
    if "docker" in bin:
        if pathlib_path("/usr/local/bin/docker").exists():
            bin_name = local["/usr/local/bin/docker"]
        elif pathlib_path("/usr/bin/docker").exists():
            bin_name = local["/usr/bin/docker"]
        else:
            raise ManagerInvalidShellCmd("Can't find docker executable!")
    else:
        bin_name = local[bin]
    p = bin_name.popen(
        args=args, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for _ in io.TextIOWrapper(p.stdout, encoding="utf-8"):
        if log_output:
            logger.info(_)
        stdout += _
    for _ in io.TextIOWrapper(p.stderr, encoding="utf-8"):
        if log_output:
            logger.error(_)
        stderr += _
    p.communicate()
    return Output(
        returncode=p.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run(bin: str, *args: str, verbose: bool = True, **kwargs: t.Any) -> None:
    result = shellcmd(bin, *args, **kwargs)
    if result.returncode > 0:
        logger.error(result.stderr)
    if verbose and result.stdout != "":
        logger.info(result.stdout)


def graceful_exit(func: "t.Callable[P, t.Any]") -> "t.Callable[P, t.Any]":
    @wraps(func)
    def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> "t.Any":
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info("Stopping now...")
            atexit.register(run, "docker", "buildx", "prune", log_output=False)
            sys.exit(0)

    return wrapper


def raise_exception(func: "t.Callable[P, t.Any]") -> "t.Callable[P, t.Any]":
    @wraps(func)
    def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> "t.Any":
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            raise ManagerException from err

    return wrapper
