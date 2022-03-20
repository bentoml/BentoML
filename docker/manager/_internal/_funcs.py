from __future__ import annotations

import os
import re
import sys
import typing as t
import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path
from functools import wraps

import cattr
from jinja2 import Environment
from fs.base import FS
from simple_di import inject
from simple_di import Provide
from jinja2.loaders import FileSystemLoader
from toolz.dicttoolz import keyfilter
from python_on_whales import docker
from python_on_whales.exceptions import DockerException
from python_on_whales.components.buildx.cli_wrapper import Builder

from .exceptions import ManagerException
from ._configuration import DockerManagerContainer

if TYPE_CHECKING:

    P = t.ParamSpec("P")
    GenericDict = t.Dict[str, t.Any]
    GenericFunc = t.Callable[P, t.Any]

logger = logging.getLogger(__name__)


def sprint(*args: t.Any, **kwargs: t.Any) -> None:
    # stream logs inside docker container to sys.stderr
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    try:
        args = tuple(map(lambda x: re.sub(r"[\(\[].*?[\)\]]", "", x), args))
    except Exception:
        pass
    print(*args, file=sys.stderr, flush=True, sep=sep, end=end)


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


cattr.register_unstructure_hook(Path, lambda x: str(x.__fspath__()))
cattr.register_unstructure_hook(FS, lambda x: repr(x))


def raise_exception(func: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            raise ManagerException from err

    return wrapper


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


def preprocess_template_paths(input_name: str, arch: t.Optional[str]) -> str:
    if "readme" in input_name.lower():
        return "README.md"
    elif "conda" in input_name:
        return "Dockerfile-conda"
    elif "-dockerfile.j2" in input_name:
        return "Dockerfile"
    else:
        input_name = input_name.split("/")[-1]
        # specific cases for rhel
        output_name = (
            input_name[len("cudnn-") : -len(".j2")]
            if input_name.startswith("cudnn-") and input_name.endswith(".j2")
            else input_name
        )
        if arch:
            output_name += f"-{arch}"
        return output_name


def _contains_key(
    filter: t.List[t.Any],
    mapping: GenericDict,
) -> GenericDict:
    return keyfilter(lambda x: x in filter, mapping)


CUSTOM_FUNCTION: t.Dict[str, GenericFunc[...]] = {"contains_key": _contains_key}


@inject
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
    fs: FS = Provide[DockerManagerContainer.root_fs],
    **kwargs: t.Any,
) -> None:

    if output_name is not None:
        output_name_ = output_name
    else:
        output_name_ = preprocess_template_paths(input_name, arch)

    out_path_fs = out_fs.makedirs(output_path, recreate=True)

    docker_loader = FileSystemLoader(fs.getsyspath("/"), followlinks=True)

    template_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=docker_loader,
    )

    if custom_function is not None:
        CUSTOM_FUNCTION.update(custom_function)
    template_env.globals.update(CUSTOM_FUNCTION)

    with inp_fs.open(input_name, "r") as inf:
        template = template_env.from_string(inf.read())

    content = template.render(build_tag=build_tag, **kwargs)
    out_path_fs.writetext(output_name_, content, newline="\n")
    out_path_fs.close()
