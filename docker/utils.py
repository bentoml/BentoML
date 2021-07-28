import errno
import json
import logging
import operator
import os
import pathlib
import string
import sys
import typing as t
from functools import reduce
from types import FunctionType

from absl import flags
from cerberus import Validator
from glom import Assign, Path, PathAccessError, PathAssignError, glom
from ruamel import yaml

__all__ = (
    "FLAGS",
    "cached_property",
    "ColoredFormatter",
    "mkdir_p",
    "flatten",
    "mapfunc",
    "maxkeys",
    "walk",
    "load_manifest_yaml",
    "sprint",
    "jprint",
    "pprint",
    "get_data",
    "set_data",
    "get_nested",
)

log = logging.getLogger(__name__)

SUPPORTED_OS = ["debian10", "centos8", "centos7", "alpine3.14", "amazonlinux2"]
SUPPORTED_GENERATE_TYPE = ["dockerfiles", "images"]

ValidateType = t.Union[str, t.List[str]]

# defined global vars.
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

flags.DEFINE_string(
    "bentoml_version", None, "BentoML release version", required=True, short_name="bv"
)
flags.DEFINE_string("cuda_version", "11.3.1", "Define CUDA version", short_name="cv")
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
flags.DEFINE_boolean(
    "push", False, "Whether to push built image.", short_name="p",
)
flags.DEFINE_boolean(
    "readmes", False, "Whether to stop at pushing readmes.", short_name="md",
)
flags.DEFINE_string(
    "releases",
    "",
    "Set of releases to build and tags, defaults to every release type.",
    short_name="r",
)

SPEC_SCHEMA = r"""
---
specs:
  required: true
  type: dict
  schema:
    releases:
      required: true
      type: dict
      schema:
        templates_dir:
          type: string
          nullable: False
        base_image:
          type: string
          nullable: False
        add_to_tags:
          type: string
          nullable: False
          required: true
        multistage_image:
          type: boolean
        header:
          type: string
        envars:
          type: list
          forbidden: ['HOME']
          default: []
          schema:
            check_with: args_format
            type: string
        cuda_prefix_url:
          type: string
          dependencies: cuda
        cuda_requires:
          type: string
          dependencies: cuda
        cuda:
          type: dict
          matched: 'specs.dependencies.cuda'
    tag:
      required: True
      type: dict
      schema:
        fmt:
          type: string
          check_with: keys_format
          regex: '({(.*?)})'
        release_type:
          type: string
          nullable: True
        python_version:
          type: string
          nullable: True
        suffixes:
          type: string
          nullable: True
    dependencies:
      type: dict
      keysrules:
        type: string
      valuesrules:
        type: dict
        keysrules:
          type: string
        valuesrules:
          type: string
          nullable: true
    repository:
      type: dict
      schema:
        pwd:
          env_vars: true
          type: string
        user:
          dependencies: pwd
          env_vars: true
          type: string
        urls:
          keysrules:
            type: string
          valuesrules:
            type: string
            regex: '((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w\-_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)'
        registry:
          keysrules:
            type: string
          type: dict
          valuesrules:
            type: string

repository:
  type: dict
  keysrules:
    type: string
    regex: '(\w+)\.{1}(\w+)'
  valuesrules:
    type: dict
    matched: 'specs.repository'

cuda:
  type: dict
  nullable: false
  keysrules:
    type: string
    required: true
    regex: '(\d{1,2}\.\d{1}\.\d)?$'
  valuesrules:
    type: dict
    keysrules:
      type: string
      check_with: cudnn_threshold
    valuesrules:
      type: string

packages:
  type: dict
  keysrules:
    type: string
  valuesrules:
    type: dict
    allowed: [devel, cudnn, runtime]
    schema:
      devel:
        type: list
        dependencies: runtime
        check_with: available_dists
        schema:
          type: string
      runtime:
        required: true
        type: [string, list]
        check_with: available_dists
      cudnn:
        type: [string, list]
        check_with: available_dists
        dependencies: runtime

releases:
  type: dict
  keysrules:
    supported_dists: true
    type: string
  valuesrules:
    type: dict
    required: True
    matched: 'specs.releases'
"""  # noqa: W605, E501 # pylint: disable=W1401


class MetadataSpecValidator(Validator):
    """
    Custom cerberus validator for BentoML metadata spec.

    Refers to https://docs.python-cerberus.org/en/stable/customize.html#custom-rules.
    This will add two rules:
    - args should have the correct format: ARG=foobar as well as correct ENV format.
    - releases should be defined under SUPPORTED_OS

    Args:
        packages: bentoml release packages, model-server, yatai-service, etc
    """

    CUDNN_THRESHOLD: int = 1
    CUDNN_COUNTER: int = 0

    def __init__(self, *args: str, **kwargs: str) -> None:
        if "packages" in kwargs:
            self.packages = kwargs["packages"]
        if "releases" in kwargs:
            self.releases = kwargs["releases"]
        super(MetadataSpecValidator, self).__init__(*args, **kwargs)

    def _check_with_packages(self, field: str, value: str) -> None:
        """
        Check if registry is defined in packages.
        """
        if value not in self.packages:
            self._error(field, f"{field} is not defined under packages")

    def _check_with_args_format(self, field: str, value: ValidateType) -> None:
        """
        Check if value has correct envars format: ARG=foobar
        This will be parsed at runtime when building docker images.
        """
        if isinstance(value, str):
            if "=" not in value:
                self._error(
                    field, f"{value} should have format ARG=foobar",
                )
            else:
                envars, _, _ = value.partition("=")
                self._validate_env_vars(True, field, envars)
        if isinstance(value, list):
            for v in value:
                self._check_with_args_format(field, v)

    def _check_with_keys_format(self, field: str, value: str) -> None:
        fmt_field = [t[1] for t in string.Formatter().parse(value) if t[1] is not None]
        for _keys in self.document.keys():
            if _keys == field:
                continue
            if _keys not in fmt_field:
                self._error(
                    field, f"{value} doesn't contain {_keys} in defined document scope"
                )

    def _check_with_cudnn_threshold(self, field: str, value: str) -> None:
        if "cudnn" in value:
            self.CUDNN_COUNTER += 1
        if self.CUDNN_COUNTER > self.CUDNN_THRESHOLD:
            self._error(field, "Only allowed one CUDNN version per CUDA mapping")

    def _check_with_available_dists(self, field: str, value: ValidateType) -> None:
        if isinstance(value, list):
            for v in value:
                self._check_with_available_dists(field, v)
        else:
            if value not in self.releases:
                self._error(field, f"{field} is not defined under releases")

    def _validate_supported_dists(
        self, supported_dists: bool, field: str, value: ValidateType
    ) -> None:
        """
        Validate if given is a supported OS.

        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if isinstance(value, str):
            if supported_dists and value not in SUPPORTED_OS:
                self._error(
                    field,
                    f"{value} is not defined in SUPPORTED_OS."
                    "If you are adding a new distros make sure "
                    "to add it to SUPPORTED_OS",
                )
        if isinstance(value, list):
            for v in value:
                self._validate_supported_dists(supported_dists, field, v)

    def _validate_env_vars(self, env_vars: bool, field: str, value: str) -> None:
        """
        Validate if given is a environment variable.

        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if isinstance(value, str):
            if env_vars and not value.isupper():
                self._error(field, f"{value} cannot be parsed to envars.")
        else:
            self._error(field, f"{value} cannot be parsed as string.")

    def _validate_matched(
        self,
        others: t.Union[ValidateType, t.Dict[str, t.Any]],
        field: str,
        value: t.Union[ValidateType, t.Dict[str, t.Any]],
    ) -> None:
        """
        Validate if a field is defined as a reference to match all given fields if
        we updated the default reference.
        The rule's arguments are validated against this schema:
            {'type': 'string'}
        """
        if isinstance(others, dict):
            if len(others) != len(value):
                self._error(
                    field,
                    f"type {type(others)} with ref "
                    f"{field} from {others} does not match",
                )
        elif isinstance(others, str):
            ref = get_nested(self.root_document, others.split("."))
            if len(value) != len(ref):
                self._error(field, f"Reference {field} from {others} does not match")
        elif isinstance(others, list):
            for v in others:
                self._validate_matched(v, field, value)
        else:
            self._error(field, f"Unable to parse field type {type(others)}")


class _Missing(object):
    def __repr__(self) -> str:
        return "no value"

    def __reduce__(self) -> str:
        return "_missing"


class cached_property(property):
    """Direct ports from bentoml/utils/__init__.py"""

    _missing = _Missing()

    def __init__(
        self,
        func: FunctionType,
        name: t.Optional[str] = None,
        doc: t.Optional[str] = None,
    ) -> None:
        super(cached_property, self).__init__(doc=doc)
        self.__name__ = name or func.__name__
        self.__doc__ = doc or func.__doc__
        self.__module__ = func.__module__
        self.func = func

    def __set__(self, obj, value):  # type: ignore
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, types=None):  # type: ignore
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    blue: str = "\x1b[34m"
    lightblue: str = "\x1b[36m"
    yellow: str = "\x1b[33m"
    red: str = "\x1b[31m"
    reset: str = "\x1b[0m"
    _format: str = "[%(levelname)s::L%(lineno)d] %(message)s"

    FORMATS = {
        logging.INFO: blue + _format + reset,
        logging.DEBUG: lightblue + _format + reset,
        logging.WARNING: yellow + _format + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: red + _format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def mkdir_p(path: str) -> None:
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def flatten(arr: t.Iterable) -> t.Generator:  # TODO: better type hint
    for it in arr:
        if isinstance(it, t.Iterable) and not isinstance(it, str):
            yield from flatten(it)
        else:
            yield it


def mapfunc(func: t.Callable, obj: t.Union[t.MutableMapping, list]) -> t.Any:
    if isinstance(obj, list):
        return func(obj)
    if any(isinstance(v, dict) for v in obj):
        for v in obj:
            mapfunc(func, v)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, t.MutableMapping):
                mapfunc(func, v)
            elif not isinstance(v, str):
                obj[k] = func(v)
            continue


def maxkeys(di: dict) -> t.Any:
    if any(isinstance(v, dict) for v in di.values()):
        for v in di.values():
            return maxkeys(v)
    return max(di.items(), key=lambda x: len(set(x[0])))[1]


def walk(path: pathlib.Path) -> t.Generator:
    for p in path.iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p.resolve()


def sprint(*args: t.Any, **kwargs: str) -> None:
    # stream logs inside docker container to sys.stderr
    print(*args, file=sys.stderr, flush=True, **kwargs)


def jprint(*args: str, **kwargs: str) -> None:
    sprint(json.dumps(args[0], indent=2), *args[1:], **kwargs)


def pprint(*args: str) -> None:
    # custom pretty logs
    log.warning(f"{'-' * 59}\n\t{' ' * 8}| {''.join([*args])}\n\t{' ' * 8}{'-' * 59}")


def get_data(obj: t.Union[dict, t.MutableMapping], *path: str) -> t.Any:
    """
    Get data from the object by dotted path.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        data = glom(obj, Path(*path))
    except PathAccessError:
        log.exception(
            "Exception occurred in "
            "get_data, unable to retrieve "
            f"{'.'.join([*path])} from {repr(obj)}"
        )
        exit(1)
    else:
        return data


def set_data(obj: dict, value: t.Union[dict, list, str], *path: str) -> None:
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(Path(*path), value))
    except PathAssignError:
        log.exception(
            "Exception occurred in "
            "update_data, unable to update "
            f"{Path(*path)} with {value}"
        )
        exit(1)


def get_nested(obj: t.Dict, keys: t.List[str]) -> t.Any:
    """Iterate through a nested dict from a list of keys"""
    return reduce(operator.getitem, keys, obj)


def load_manifest_yaml(file: str) -> dict:
    with open(file, "r") as input_file:
        manifest: t.Dict = yaml.safe_load(input_file)

    v: MetadataSpecValidator = MetadataSpecValidator(
        yaml.safe_load(SPEC_SCHEMA),
        packages=manifest["packages"],
        releases=manifest["releases"],
    )

    if not v.validate(manifest):
        v.clear_caches()
        log.error(f"{file} is invalid. Errors as follow:")
        sprint(yaml.dump(v.errors, indent=2))
        exit(1)
    else:
        log.info(f"Valid {file}.")

    normalized: dict = v.normalized(manifest)
    return normalized
