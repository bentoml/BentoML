import errno
import json
import logging
import os
import pathlib
import string
import sys
from typing import Callable, MutableMapping, Iterable, Dict, List, Union, Generator

from absl import flags
from cerberus import Validator
from glom import glom, Path, PathAccessError, Assign, PathAssignError
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
)

log = logging.getLogger(__name__)

SUPPORTED_OS = ["debian10", "centos8", "centos7", "alpine3.14", "amazonlinux2"]
SUPPORTED_GENERATE_TYPE = ['dockerfiles', 'images']

# defined global vars.
FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "push_to_hub",
    False,
    "Whether to upload images to given registries.",
    short_name="pth",
)
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

flags.DEFINE_string(
    "releases",
    "",
    "Set of releases to build and tags, defaults to every release type.",
    short_name='r',
)

SPEC_SCHEMA = """
---
repository:
  type: dict
  keysrules:
    type: string
    regex: '(\w+)\.{1}(\w+)'
  valuesrules:
    type: dict
    schema:
      pwd:
        env_vars: true
        type: string
      user:
        dependencies: pwd
        env_vars: true
        type: string
      registry:
        keysrules:
          check_with: packages
          type: string
        type: dict
        valuesrules:
          type: string

cuda_dependencies:
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
      check_with: allowed_cudnn
    valuesrules:
      type: string

release_spec:
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
      match_ref: 'cuda_dependencies'

tag_spec:
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

packages:
  type: dict
  keysrules:
    type: string
    check_with: packages
  valuesrules:
    type: dict
    allowed: [devel, cudnn, runtime]
    schema:
      devel:
        type: list
        dependencies: runtime
        check_with: dists_defined
        schema:
          type: string
      runtime:
        required: true
        type: [string, list]
        check_with: dists_defined
      cudnn:
        type: [string, list]
        check_with: dists_defined
        dependencies: runtime

releases:
  type: dict
  keysrules:
    supported_dists: true
    type: string
  valuesrules:
    type: dict
    required: True
    match_ref: 'release_spec'
"""


# TODO: Type annotation.
# class Field(Any):


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
    cudnn_counter: int = 0

    # There are three publics cls objects we can access
    # v.schema, v.root_document, v.document -> normalized root_document

    def __init__(self, *args, **kwargs):
        if "packages" in kwargs:
            self.packages = kwargs["packages"]
        if "releases" in kwargs:
            self.releases = kwargs["releases"]
        super(Validator, self).__init__(*args, **kwargs)

    def _check_with_keys_format(self, field, value):
        fmt_field = [t[1] for t in string.Formatter().parse(value) if t[1] is not None]
        for _keys in self.document.keys():
            if _keys == field:
                continue
            if _keys not in fmt_field:
                self._error(
                    field, f"{value} doesn't contain {_keys} in defined document scope"
                )

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

    def _check_with_dists_defined(self, field, value):
        if isinstance(value, list):
            for v in value:
                self._check_with_dists_defined(field, v)
        else:
            if value not in self.releases:
                self._error(field, f"{field} is not defined under releases")

    def _check_with_allowed_cudnn(self, field, value):
        if 'cudnn' in value:
            self.cudnn_counter += 1
            pass
        if self.cudnn_counter > self.CUDNN_THRESHOLD:
            self._error(field, "Only allowed one CUDNN version per CUDA mapping")

    def _check_with_format_keys(self, field, value):
        pass

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

    def _validate_match_ref(self, others, field, value):
        """
        Validate if a field is defined as a reference to match all given fields if
        we updated the default reference.
        The rule's arguments are validated against this schema:
            {'type': 'string'}
        """
        try:
            if others not in self.root_document:
                return False
            ref = self.root_document[others]
            if any(isinstance(v, dict) for v in ref.values()):
                for v in ref.values():
                    return self._validate_match_ref(v, field, value)
            if len(value) != len(ref):
                self._error(field, f"Reference {field} from {others} does not match")
        except TypeError:
            # when others is a dictionary.
            if len(others) != len(value):
                self._error(field, f"Reference {field} from {others} does not match")


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


def flatten(arr: Iterable) -> Iterable[List]:
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
    log.warning(f"{'-' * 59}\n\t{' ' * 8}| {''.join(*args)}\n\t{' ' * 8}{'-' * 59}")


def get_data(obj: Union[Dict, MutableMapping], *path: str) -> Union[str, Dict, List]:
    """
    Get data from the object by dotted path.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        data = glom(obj, Path(*path))
    except PathAccessError:
        log.exception(
            f"Exception occurred in get_data, unable to retrieve {'.'.join([*path])} from {repr(obj)}"
        )
        exit(1)
    else:
        return data


def set_data(obj: Dict, value: Union[Dict, str], *path: str):
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(Path(*path), value))
    except ValueError:
        # we can just use dict.update to handle this error.
        obj.update(value)
    except PathAssignError:
        log.exception(
            f"Exception occurred in update_data, unable to update {Path(*path)} with {value}"
        )
        exit(1)


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
