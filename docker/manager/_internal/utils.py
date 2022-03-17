from __future__ import annotations

import os
import re
import sys
import json
import typing as t
import logging
import subprocess
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path as pathlib_path
from functools import wraps
from functools import partial

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
from toolz.dicttoolz import keyfilter
from python_on_whales import docker
from python_on_whales.exceptions import DockerException
from python_on_whales.components.buildx.cli_wrapper import Builder

from .exceptions import ManagerException
from .exceptions import ManagerInvalidShellCmd

if TYPE_CHECKING:
    from plumbum.machines.local import LocalCommand
    from plumbum.machines.local import PlumbumLocalPopen

    from .types import P
    from .types import GenericDict
    from .types import GenericFunc
    from .types import GenericList

logger = logging.getLogger(__name__)

SUPPORTED_PYTHON_VERSION = ("3.6", "3.7", "3.8", "3.9", "3.10")
SUPPORTED_REGISTRIES = ("docker.io", "ecr", "quay.io", "gcr")
DOCKERFILE_BUILD_HIERARCHY = ("base", "runtime", "cudnn", "devel")

SUPPORTED_ARCHITECTURE_TYPE = ["amd64", "arm64v8", "ppc64le", "s390x"]

# file
EXTENSION = ".j2"
DOCKERFILE_NAME = "Dockerfile"
DOCKERFILE_TEMPLATE_SUFFIX = f"-{DOCKERFILE_NAME.lower()}{EXTENSION}"


def send_log(*args: t.Any, _manager_level: int = logging.INFO, **kwargs: t.Any):
    if os.path.exists("/.dockerenv"):
        sprint(*args, **kwargs)
    else:
        log_by_level = {
            logging.ERROR: logger.error,
            logging.INFO: logger.info,
            logging.WARNING: logger.warning,
            logging.DEBUG: logger.debug,
            logging.NOTSET: logger.error,
            logging.CRITICAL: logger.critical,
        }[_manager_level]
        log_by_level(*args, **kwargs)


def stream_docker_logs(resp: t.Iterator[str], tag: str) -> None:
    while True:
        try:
            output = next(resp)
            send_log(f"[{tag.split('/')[-1]}] {output}")
        except DockerException as e:
            message = f"[{type(e).__name__}] Error while streaming logs:\n{e}"
            send_log(message, _manager_level=logging.ERROR)
            break
        except StopIteration:
            send_log(f"Successfully built {tag}")
            break


def serializer(inst: type, field: attrs.Attribute[t.Any], value: t.Any) -> t.Any:
    if isinstance(value, pathlib_path):
        return value.as_posix()
    elif isinstance(value, FS):
        return repr(value)
    else:
        return value


def is_attrs_cls(cls: type):
    return hasattr(cls, "__attrs_attrs__")


def create_buildx_builder(name: str, verbose_: bool = False) -> Builder:
    buildx_list = [i.name for i in docker.buildx.list()]
    if name not in buildx_list:
        builder = docker.buildx.create(
            use=True,
            name=name,
            driver="docker-container",
            driver_options={"image": "moby/buildkit:master"},
        )
        if verbose_:
            docker.buildx.inspect(name)
        return builder
    else:
        return [i for i in docker.buildx.list() if i.name == name][0]


def ctx_unstructure_hook(
    ctx: t.Optional[type],
) -> t.Optional[t.Dict[str, t.Any]]:
    if not ctx or not is_attrs_cls(ctx):
        return None
    return attrs.asdict(ctx, value_serializer=serializer)


def unstructure(dct: t.Dict[str, t.List[t.Any]]) -> t.Dict[str, t.Any]:
    return {
        key: [cattr.unstructure(_) if is_attrs_cls(_) else None for _ in value]
        for key, value in dct.items()
    }


def graceful_exit(func: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            send_log("Stopping now...")
            sys.exit(0)

    return wrapper


def raise_exception(func: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            raise ManagerException from err

    return wrapper


def as_posix(*args: t.Any) -> str:
    return pathlib_path(*args).as_posix()


def inject_deepcopy_args(
    _func: t.Optional[t.Callable[P, t.Any]] = None, *, num_args: int = 1
) -> t.Callable[P, t.Any]:
    def decorator(func: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            args = tuple(deepcopy(a) for a in args[:num_args]) + args[num_args:]  # type: ignore
            return func(*args, **kwargs)

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def sprint(*args: t.Any, **kwargs: t.Any) -> None:
    # stream logs inside docker container to sys.stderr
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    args = tuple(map(lambda x: re.sub(r"[\(\[].*?[\)\]]", "", x), args))
    print(*args, file=sys.stderr, flush=True, sep=sep, end=end)


def jprint(*args: t.Any, indent: int = 2, **kwargs: t.Any) -> None:
    sprint(json.dumps(args[0], indent=indent), *args[1:], **kwargs)


def process_docker_arch(arch: str) -> str:
    # NOTE: hard code these platform, seems easy to manage
    if "arm64" in arch:
        fmt = "arm64/v8"
    elif "arm32" in arch:
        fmt = "/".join(arch.split("32"))
    elif "i386" in arch:
        fmt = arch[1:]
    else:
        fmt = arch
    return fmt


def get_docker_platform_mapping() -> t.Dict[str, str]:
    return {k: f"linux/{process_docker_arch(k)}" for k in SUPPORTED_ARCHITECTURE_TYPE}


def get_data(
    obj: t.Union[GenericDict, t.MutableMapping[str, t.Any]], *path: str
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
    obj: GenericDict, value: t.Union[GenericDict, GenericList, str], *path: str
) -> None:
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(glom_path(*path), value))
    except PathAssignError as err:
        send_log(
            "Exception occurred in "
            "update_data, unable to update "
            f"{glom_path(*path)} with {value}",
            _manager_level=logging.ERROR,
        )
        raise ManagerException("Error while setting data to object.") from err


@attrs.define
class Output:
    returncode: int
    stderr: str
    stdout: str


def get_bin(bin: str) -> t.Tuple[LocalCommand, str]:
    if "docker" in bin:
        if pathlib_path("/usr/local/bin/docker").exists():
            name = "/usr/local/bin/docker"
        elif pathlib_path("/usr/bin/docker").exists():
            name = "/usr/bin/docker"
        else:
            raise ManagerInvalidShellCmd("Can't find docker executable!")
    else:
        name = bin

    return local[name], name


def shellcmd(
    bin: str,
    *args: str,
    shell: bool = False,
    return_proc: bool = False,
) -> "t.Union[partial[PlumbumLocalPopen], Output]":
    bin_name, _ = get_bin(bin)
    proc = partial(
        bin_name.popen,
        args=args,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if return_proc:
        return proc
    else:
        p = proc()
        stdout, stderr = p.communicate()
        return Output(
            returncode=p.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def run(
    bin: str, *args: str, return_proc: bool = False, **kwargs: t.Any
) -> "t.Union[int, partial[PlumbumLocalPopen]]":
    p = shellcmd(bin, *args, return_proc=return_proc, **kwargs)
    if isinstance(p, Output):
        return p.returncode
    else:
        return p


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


def _contains_key(
    filter: GenericList,
    mapping: GenericDict,
) -> GenericDict:
    return keyfilter(lambda x: x in filter, mapping)


CUSTOM_FUNCTION = {"contains_key": _contains_key}


def render_template(
    input_name: str,
    inp_fs: FS,
    output_path: str,
    out_fs: FS,
    *,
    output_name: t.Optional[str] = None,
    arch: t.Optional[str] = None,
    build_tag: t.Optional[str] = None,
    custom_function: t.Optional[t.Dict[str, GenericFunc[t.Any]]] = None,
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
    if custom_function is not None:
        CUSTOM_FUNCTION.update(custom_function)  # type: ignore
    template_env.globals.update(CUSTOM_FUNCTION)

    with inp_fs.open(input_name, "r") as inf:
        template = template_env.from_string(inf.read())

    out_path_fs.writetext(
        output_name_, template.render(build_tag=build_tag, **kwargs), newline="\n"
    )
    out_path_fs.close()
