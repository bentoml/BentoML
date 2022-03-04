import io
import os
import re
import sys
import json
import typing as t
import pathlib
import operator
import subprocess
from typing import TYPE_CHECKING
from functools import wraps
from functools import reduce

import attrs
from log import logger
from absl import flags
from glom import glom
from glom import Path as glom_path
from glom import Assign
from glom import PathAccessError
from glom import PathAssignError
from ruamel import yaml
from plumbum import local
from validator import SPEC_SCHEMA
from validator import MetadataSpecValidator
from exceptions import InvalidShellCommand

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    DictStrAny = t.Dict[str, t.Any]
    DictStrStr = t.Dict[str, str]
    GenericFunc = t.Callable[P, t.Any]

__all__ = (
    "FLAGS",
    "load_manifest_yaml",
    "sprint",
    "jprint",
    "get_data",
    "walk",
    "set_data",
    "get_nested",
    "shellcmd",
    "graceful_exit",
    "as_posix",
)


# TODO: look into Python 3.10 installation issue with conda
SUPPORTED_PYTHON_VERSION: t.List[str] = ["3.7", "3.8", "3.9"]
SUPPORTED_GENERATE_TYPE = ["dockerfiles", "images"]
SUPPORTED_ARCHITECTURE_TYPE = [
    "amd64",
    "arm32v5",
    "arm32v6",
    "arm32v7",
    "arm64v8",
    "i386",
    "ppc64le",
    "s390x",
    "riscv64",
]
DOCKERFILE_BUILD_HIERARCHY = ("base", "runtime", "cudnn", "devel")


FLAGS = flags.FLAGS

flags.DEFINE_integer("timeout", 3600, "Timeout for docker build", short_name="t")
# CLI-related
flags.DEFINE_boolean(
    "dump_metadata",
    False,
    "Dump all tags metadata to file and stops if specified.",
    short_name="dm",
)
flags.DEFINE_boolean(
    "dry_run",
    False,
    "Whether to dry run. This won't create Dockerfile.",
    short_name="dr",
)
flags.DEFINE_boolean("overwrite", False, "Overwrite built images.", short_name="o")
flags.DEFINE_boolean("validate", False, "Stop at manifest validation", short_name="vl")

# directory and files
flags.DEFINE_string(
    "dockerfile_dir",
    "./generated",
    "path to generated Dockerfile. Existing files will be deleted with new Dockerfiles",
    short_name="dd",
)
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file", short_name="mf")

# versions
flags.DEFINE_string(
    "bentoml_version", None, "BentoML release version", required=True, short_name="bv"
)
flags.DEFINE_string("cuda_version", "11.5.1", "Define CUDA version", short_name="cv")
flags.DEFINE_multi_string(
    "python_version",
    [],
    "OPTIONAL: Python version to build Docker images (useful when developing).",
    short_name="pv",
)
flags.DEFINE_multi_string(
    "generate",
    [],
    f"Generation type. Options: {SUPPORTED_GENERATE_TYPE}",
    short_name="g",
)
flags.DEFINE_multi_string(
    "arch",
    [],
    f"Supported architecture. Options: {SUPPORTED_ARCHITECTURE_TYPE}",
    short_name="a",
)
flags.DEFINE_boolean(
    "push",
    False,
    "Whether to push built image.",
    short_name="p",
)
flags.DEFINE_boolean(
    "readmes",
    False,
    "Whether to stop at pushing readmes.",
    short_name="md",
)
flags.DEFINE_string(
    "releases",
    "",
    "Set of releases to build and tags, defaults to every release type.",
    short_name="r",
)


def as_posix(*args: t.Any) -> str:
    return pathlib.Path(*args).as_posix()


def walk(
    path: pathlib.Path, exclude: t.Optional[t.Tuple[str]] = None
) -> t.Generator[pathlib.Path, None, None]:
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


def get_data(
    obj: "t.Union[DictStrAny, t.MutableMapping[str, t.Any]]", *path: str
) -> "DictStrAny":
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


def set_data(
    obj: "DictStrAny", value: "t.Union[DictStrAny, t.List[t.Any], str]", *path: str
) -> None:
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(glom_path(*path), value))
    except PathAssignError:
        logger.exception(
            "Exception occurred in "
            "update_data, unable to update "
            f"{glom_path(*path)} with {value}"
        )
        exit(1)


def get_nested(obj: "DictStrAny", *keys: str) -> t.Any:
    """Iterate through a nested dict from a list of keys"""
    return reduce(operator.getitem, [keys], obj)


def load_manifest_yaml(file: str, validate: bool = False) -> "DictStrAny":
    with open(file, "r", encoding="utf-8") as input_file:
        manifest = yaml.safe_load(input_file)

    if not validate:
        return manifest
    else:
        # TODO: validate manifest.yml
        # Necessary for CI pipelines
        # Parse specs from manifest.yml and validate it.
        v = MetadataSpecValidator(
            yaml.safe_load(SPEC_SCHEMA),
            packages=manifest["packages"],
            releases=manifest["releases"],
        )
        if not v.validate(manifest):
            v.clear_caches()
            logger.error(f"{file} is invalid. Errors as follow:")
            sprint(yaml.dump(v.errors, indent=2))
            exit(1)
        else:
            logger.info(f"Valid {file}.")

        return v.normalized(manifest)


@attrs.define
class Output:
    returncode: int
    stderr: str
    stdout: str


def shellcmd(bin: str, *args: str) -> Output:
    stdout, stderr = "", ""
    if "docker" in bin:
        if pathlib.Path("/usr/local/bin/docker").exists():
            bin_name = local["/usr/local/bin/docker"]
        elif pathlib.Path("/usr/bin/docker").exists():
            bin_name = local["/usr/bin/docker"]
        else:
            raise InvalidShellCommand("Can't find docker client executable!")
    else:
        bin_name = local[bin]
    p = bin_name.popen(
        args=args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for l in io.TextIOWrapper(p.stdout, encoding="utf-8"):
        logger.info(l)
        stdout += l
    for l in io.TextIOWrapper(p.sterr, encoding="utf-8"):
        logger.error(l)
        stderr += l
    return Output(
        returncode=p.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def graceful_exit(func: "t.Callable[P, t.Any]") -> "t.Callable[P, t.Any]":
    @wraps(func)
    def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info("\nStopping now...")
            sys.exit(1)

    return wrapper


def argv_validator(length: int = 1) -> "t.Callable[[GenericFunc], GenericFunc]":
    def sanity_check(func: "GenericFunc") -> "GenericFunc":
        @wraps(func)
        def wrapper(argv: str, *args: "P.args", **kwargs: "P.kwargs") -> t.Any:
            if len(argv) > length:
                raise RuntimeError("Too much arguments")

            # validate releases args.
            if FLAGS.releases and FLAGS.releases not in DOCKERFILE_BUILD_HIERARCHY:
                logger.critical(
                    f"Invalid --releases arguments. Allowed: {DOCKERFILE_BUILD_HIERARCHY}"
                )
            if len(FLAGS.arch) > 0:
                logger.debug("--arch defined, ignoring SUPPORTED_ARCHITECTURE_TYPE")
                supported_arch_type = FLAGS.arch
                if supported_arch_type not in SUPPORTED_ARCHITECTURE_TYPE:
                    logger.critical(
                        f"Invalid --arch arguments. Allowed: {SUPPORTED_ARCHITECTURE_TYPE}"
                    )
            else:
                supported_arch_type = SUPPORTED_ARCHITECTURE_TYPE

            # injected supported_python_version
            if len(FLAGS.python_version) > 0:
                logger.debug(
                    "--python_version defined, ignoring SUPPORTED_PYTHON_VERSION"
                )
                supported_python_version = FLAGS.python_version
                if supported_python_version not in SUPPORTED_PYTHON_VERSION:
                    logger.critical(
                        f"{supported_python_version} is not supported. Allowed: {SUPPORTED_PYTHON_VERSION}"
                    )
            else:
                supported_python_version = SUPPORTED_PYTHON_VERSION

            # Refers to manifest.yml under field packages and releases
            kwargs.update(
                {
                    "supported_python_version": supported_python_version,
                    "supported_arch_type": supported_arch_type,
                }
            )
            return func(*args, **kwargs)

        return wrapper

    return sanity_check
