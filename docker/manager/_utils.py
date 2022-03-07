import io
import sys
import json
import atexit
import typing as t
import logging
import traceback
import subprocess
from copy import deepcopy
from typing import overload
from typing import TYPE_CHECKING
from pathlib import Path as pathlib_path
from functools import wraps

import attrs
import cattr
from glom import glom
from glom import Path as glom_path
from glom import Assign
from glom import PathAccessError
from glom import PathAssignError
from jinja2 import Environment
from fs.base import FS
from plumbum import local
from manager._exceptions import ManagerException
from manager._exceptions import ManagerInvalidShellCmd

if TYPE_CHECKING:
    from manager._types import P
    from manager._types import GenericDict
    from manager._types import GenericList
    from manager._schemas import BuildCtx
    from manager._schemas import ReleaseCtx
    from plumbum.machines.local import LocalCommand

logger = logging.getLogger(__name__)

__all__ = (
    "sprint",
    "raise_exception",
    "get_data",
    "set_data",
    "shellcmd",
    "graceful_exit",
    "as_posix",
)

SUPPORTED_PYTHON_VERSION = ("3.6", "3.7", "3.8", "3.9", "3.10")
SUPPORTED_OS_RELEASES = (
    "debian11",
    "debian10",
    "ubi8",
    "ubi7",
    "amazonlinux2",
    "alpine3.14",
)
SUPPORTED_REGISTRIES = ("docker.io", "ecr", "quay.io", "gcr")
DOCKERFILE_BUILD_HIERARCHY = ("base", "runtime", "cudnn", "devel")

SUPPORTED_ARCHITECTURE_TYPE = ["amd64", "arm64v8", "ppc64le", "s390x"]

SUPPORTED_ARCHITECTURE_TYPE_PER_DISTRO = {
    "debian11": SUPPORTED_ARCHITECTURE_TYPE,
    "debian10": SUPPORTED_ARCHITECTURE_TYPE,
    "ubi8": SUPPORTED_ARCHITECTURE_TYPE,
    "ubi7": ["amd64", "s390x", "ppc64le"],
    "alpine3.14": SUPPORTED_ARCHITECTURE_TYPE,
    "amazonlinux2": ["amd64", "arm64v8"],
}

TEMPLATES_DIR_MAPPING = {
    "debian": "debian",
    "ubi": "rhel",
    "amazonlinux": "rhel",
    "alpine": "alpine",
}

# file
EXTENSION = ".j2"
DOCKERFILE_NAME = "Dockerfile"
DOCKERFILE_TEMPLATE_SUFFIX = f"-{DOCKERFILE_NAME.lower()}{EXTENSION}"


@overload
def serializer(
    inst: type, field: "attrs.Attribute[t.Any]", value: pathlib_path
) -> t.Any:
    ...


@overload
def serializer(
    inst: type, field: "attrs.Attribute[pathlib_path]", value: pathlib_path
) -> t.Any:
    ...


def serializer(inst: type, field: "attrs.Attribute[t.Any]", value: t.Any) -> t.Any:
    if isinstance(value, pathlib_path):
        return value.as_posix()
    elif isinstance(value, FS):
        return repr(value)
    else:
        return value


def is_attrs_cls(cls: type):
    return hasattr(cls, "__attrs_attrs__")


def ctx_unstructure_hook(
    ctx: t.Optional[type],
) -> t.Optional[t.Dict[str, t.Any]]:
    if not ctx or not is_attrs_cls(ctx):
        return None
    return attrs.asdict(ctx, value_serializer=serializer)


@overload
def unstructure(dct: "t.Dict[str, t.List[BuildCtx]]") -> t.Dict[str, t.Any]:
    ...


@overload
def unstructure(dct: "t.Dict[str, t.List[ReleaseCtx]]") -> t.Dict[str, t.Any]:
    ...


def unstructure(dct: t.Dict[str, t.List[t.Any]]) -> t.Dict[str, t.Any]:
    return {
        key: [cattr.unstructure(_) if is_attrs_cls(_) else None for _ in value]
        for key, value in dct.items()
    }


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


def as_posix(*args: t.Any) -> str:
    return pathlib_path(*args).as_posix()


def inject_deepcopy_args(
    _func: "t.Optional[t.Callable[P, t.Any]]" = None, *, num_args: int = 1
) -> "t.Callable[P, t.Any]":
    def decorator(func: "t.Callable[P, t.Any]") -> "t.Callable[P, t.Any]":
        @wraps(func)
        def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
            args = tuple(deepcopy(a) for a in args[:num_args]) + args[num_args:]  # type: ignore
            return func(*args, **kwargs)

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


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


def get_bin(bin: str) -> "LocalCommand":
    if "docker" in bin:
        if pathlib_path("/usr/local/bin/docker").exists():
            bin_name = local["/usr/local/bin/docker"]
        elif pathlib_path("/usr/bin/docker").exists():
            bin_name = local["/usr/bin/docker"]
        else:
            raise ManagerInvalidShellCmd("Can't find docker executable!")
    else:
        bin_name = local[bin]
    return bin_name


@raise_exception
def shellcmd(
    bin: str, *args: str, log_output: bool = True, shell: bool = False
) -> Output:
    bin_name = get_bin(bin)
    stdout, stderr = "", ""
    p = bin_name.popen(
        args=args, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for i in io.TextIOWrapper(p.stdout, encoding="utf-8"):
        if log_output:
            sprint(i)
        stdout += i
    for i in io.TextIOWrapper(p.stderr, encoding="utf-8"):
        if log_output:
            sprint(i)
        stderr += i
    p.communicate()
    return Output(
        returncode=p.returncode,
        stdout=stdout,
        stderr=stderr,
    )


@raise_exception
@graceful_exit
def run(bin: str, *args: str, verbose: bool = True, **kwargs: t.Any) -> None:
    result = shellcmd(bin, *args, log_output=verbose, **kwargs)
    if result.returncode > 0:
        logger.error(result.stderr)
    if result.stdout != "":
        logger.info(result.stdout)


def preprocess_template_paths(input_name: str, arch: t.Optional[str]) -> str:
    if "readme" in input_name.lower():
        return "README.md"
    elif "dockerfile" in input_name:
        return DOCKERFILE_NAME
    else:
        input_name = input_name.split("/")[-1]
        # specific cases for rhel
        output_name = (
            input_name[len("cudnn-") : -len(EXTENSION)]
            if input_name.startswith("cudnn-") and input_name.endswith(EXTENSION)
            else input_name
        )
        if arch:
            output_name += f"-{arch}"
        return output_name


def render_template(
    input_name: str,
    inp_fs: "FS",
    output_path: str,
    out_fs: "FS",
    *,
    output_name: t.Optional[str] = None,
    arch: t.Optional[str] = None,
    build_tag: t.Optional[str] = None,
    **kwargs: t.Any,
) -> None:

    output_name_ = preprocess_template_paths(input_name=input_name, arch=arch)
    if output_name:
        output_name_ = output_name

    out_path_fs = out_fs.makedirs(output_path, recreate=True)

    template_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
    )

    with inp_fs.open(input_name, "r") as inf:
        template = template_env.from_string(inf.read())

    with out_path_fs.open(output_name_, "w", encoding="utf-8") as f:
        f.write(template.render(build_tag=build_tag, **kwargs))
