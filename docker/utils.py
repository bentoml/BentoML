import errno
import json
import logging
import operator
import os
import pathlib
import string
import sys
from functools import reduce
from typing import Any, Callable, Dict, Generator, Iterable, List, MutableMapping, Union

from absl import flags
from cerberus import Validator
from glom import Assign, Path, PathAccessError, PathAssignError, glom
from ruamel import yaml

__all__ = (
    'FLAGS',
    'cached_property',
    'ColoredFormatter',
    'mkdir_p',
    'flatten',
    'mapfunc',
    'maxkeys',
    'walk',
    'load_manifest_yaml',
    'sprint',
    'jprint',
    'pprint',
    'get_data',
    'set_data',
    'get_nested',
)

log = logging.getLogger(__name__)

SUPPORTED_OS = ["debian10", "centos8", "centos7", "alpine3.14", "amazonlinux2"]
SUPPORTED_GENERATE_TYPE = ['dockerfiles', 'images']

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
    short_name='g',
)
flags.DEFINE_boolean(
    "push", False, "Whether to push built image.", short_name='p',
)
flags.DEFINE_boolean(
    "readmes", False, "Whether to stop at pushing readmes.", short_name='md',
)
flags.DEFINE_string(
    "releases",
    "",
    "Set of releases to build and tags, defaults to every release type.",
    short_name='r',
)

SPEC_SCHEMA = """
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
            regex: '((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w\-_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)' # noqa: W605, E501
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
    regex: '(\w+)\.{1}(\w+)' # noqa: W605
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
"""  # noqa: E501


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

    def __init__(self, *args, **kwargs):
        if "packages" in kwargs:
            self.packages = kwargs["packages"]
        if "releases" in kwargs:
            self.releases = kwargs["releases"]
        super(Validator, self).__init__(*args, **kwargs)

    def _check_with_packages(self, field, value):
        """
        Check if registry is defined in packages.
        """
        if value not in self.packages:
            self._error(field, f"{field} is not defined under packages")

    def _check_with_args_format(self, field, value):
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

    def _check_with_keys_format(self, field, value):
        fmt_field = [t[1] for t in string.Formatter().parse(value) if t[1] is not None]
        for _keys in self.document.keys():
            if _keys == field:
                continue
            if _keys not in fmt_field:
                self._error(
                    field, f"{value} doesn't contain {_keys} in defined document scope"
                )

    def _check_with_cudnn_threshold(self, field, value):
        if 'cudnn' in value:
            self.CUDNN_COUNTER += 1
            pass
        if self.CUDNN_COUNTER > self.CUDNN_THRESHOLD:
            self._error(field, "Only allowed one CUDNN version per CUDA mapping")

    def _check_with_available_dists(self, field, value):
        if isinstance(value, list):
            for v in value:
                self._check_with_available_dists(field, v)
        else:
            if value not in self.releases:
                self._error(field, f"{field} is not defined under releases")

    def _validate_supported_dists(self, supported_dists, field, value):
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

    def _validate_env_vars(self, env_vars, field, value):
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

    def _validate_matched(self, others, field, value):
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
            ref = get_nested(self.root_document, others.split('.'))
            if len(value) != len(ref):
                self._error(field, f"Reference {field} from {others} does not match")
        elif isinstance(others, list):
            for v in others:
                self._validate_matched(v, field, value)
        else:
            self._error(field, f"Unable to parse field type {type(others)}")


class cached_property(property):
    """Direct ports from bentoml/utils/__init__.py"""

    class _Missing(object):
        def __repr__(self):
            return "no value"

        def __reduce__(self):
            return "_missing"

    _missing = _Missing()

    def __init__(self, func, name=None, doc=None):
        super().__init__(doc)
        self.__name__ = name or func.__name__
        self.__doc__ = doc or func.__doc__
        self.__module__ = func.__module__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, types=None):
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

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def mkdir_p(path: str):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def flatten(arr: Iterable) -> Iterable[Union[Any, str]]:
    for it in arr:
        if isinstance(it, Iterable) and not isinstance(it, str):
            yield from flatten(it)
        else:
            yield it


def mapfunc(func: Callable, obj: Union[MutableMapping, List]):
    if isinstance(obj, list):
        return func(obj)
    if any(isinstance(v, dict) for v in obj):
        for v in obj:
            mapfunc(func, v)
    if isinstance(obj, Dict):
        for k, v in obj.items():
            if isinstance(v, MutableMapping):
                mapfunc(func, v)
            elif not isinstance(v, str):
                obj[k] = func(v)
            continue


def maxkeys(di: Dict) -> List:
    if any(isinstance(v, dict) for v in di.values()):
        for v in di.values():
            return maxkeys(v)
    return max(di.items(), key=lambda x: len(set(x[0])))[1]


def walk(path: pathlib.Path) -> Generator:
    for p in path.iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p.resolve()


def sprint(*args, **kwargs):
    # stream logs inside docker container to sys.stderr
    print(*args, file=sys.stderr, flush=True, **kwargs)


def jprint(*args, **kwargs):
    sprint(json.dumps(*args, indent=2), **kwargs)


def pprint(*args):
    # custom pretty logs
    log.warning(f"{'-' * 59}\n\t{' ' * 8}| {''.join([*args])}\n\t{' ' * 8}{'-' * 59}")


def get_data(obj: Union[Dict, MutableMapping], *path: str) -> Any:
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


def set_data(obj: Dict, value: Union[Dict, List, str], *path: str) -> None:
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


def get_nested(obj: Dict, keys: List[str]) -> Any:
    """Iterate through a nested dict from a list of keys"""
    return reduce(operator.getitem, keys, obj)


def load_manifest_yaml(file: str) -> Dict:
    with open(file, "r") as input_file:
        manifest: Dict = yaml.safe_load(input_file)

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

    return v.normalized(manifest)
