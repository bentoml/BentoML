#!/usr/bin/env python3
import errno
import inspect
import json
import logging
import os
import pathlib
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from typing import Dict, Optional

from absl import flags, app
from cerberus import Validator
from glom import glom, Path, PathAccessError, Assign, PathAssignError
from jinja2 import Environment
from ruamel import yaml
from prettytable import PrettyTable

logging.basicConfig(level=logging.NOTSET)

# defined global vars.
FLAGS = flags.FLAGS

DOCKERFILE_PREFIX = ".Dockerfile.j2"
REPO_PREFIX = ".repo.j2"
DOCKERFILE_ALLOWED_NAMES = ['base', 'cudnn', 'devel', 'runtime']

SUPPORTED_PYTHON_VERSION = ["3.6", "3.7", "3.8"]
SUPPORTED_OS = ['ubuntu', 'debian', 'centos', 'alpine', 'amazonlinux']

NVIDIA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{}/x86_64"
NVIDIA_ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{}/x86_64"
)

flags.DEFINE_boolean(
    "push_to_hub",
    False,
    "Whether to upload images to given registries.",
    short_name='pth',
)

# CLI-related
flags.DEFINE_boolean(
    "dump_metadata",
    False,
    "Whether to dump all tags metadata to file.",
    short_name='dm',
)
flags.DEFINE_boolean(
    "dry_run",
    False,
    "Whether to dry run. This won't create Dockerfile.",
    short_name='dr',
)
flags.DEFINE_boolean(
    "stop_at_generate", False, "Whether to just generate dockerfile.", short_name='sag'
)
flags.DEFINE_boolean("build_images", False, "Whether to build images.", short_name='bi')
flags.DEFINE_boolean(
    "stop_on_failure",
    False,
    "Stop processing tags if any one build fails. If False or not specified, failures are reported but do not affect "
    "the other images.",
    short_name="sof",
)
# directory and files
flags.DEFINE_string(
    "dockerfile_dir",
    "./generated",
    "path to generated Dockerfile. Existing files will be deleted with new Dockerfiles",
    short_name="dd",
)
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file", short_name="mf")

flags.DEFINE_string(
    "bentoml_version", None, "BentoML release version", required=True, short_name='bv'
)

flags.DEFINE_string("cuda_version", "11.3.1", "Define CUDA version", short_name='cv')

flags.DEFINE_string(
    "python_version",
    None,
    "OPTIONAL: Python version to build Docker images (useful when developing).",
    short_name='pv',
)

SPEC_SCHEMA = """
--- 
repository:
  type: dict
  keysrules:
    type: string
  valuesrules:
    type: dict
    schema:
      only_if:
        required: true
        env_vars: true
        type: string
      pwd:
        dependencies: only_if
        env_vars: true
        type: string
      user:
        dependencies: only_if
        env_vars: true
        type: string
      registry:
        keysrules:
          check_with: packages
          type: string
        type: dict
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
      runtime:
        required: true
        type: [string, list]
        check_with: dists_defined
      devel:
        type: list
        dependencies: runtime
        check_with: dists_defined
        schema:
          type: string
      cudnn:
        type: [string, list]
        check_with: dists_defined
        dependencies: runtime

cuda_dependencies:
  type: dict
  nullable: false
  keysrules:
    type: string
    regex: '(\d{1,2}\.\d{1}\.\d)?$'
  valuesrules:
    type: dict
    keysrules:
      type: string
    valuesrules:
      type: string

release_spec:
  required: true
  type: dict
  schema:
    add_to_tags:
      type: string
      required: true
    templates_dir:
      type: string
    base_image:
      type: string
    header:
      type: string
    cuda_prefix_url:
      type: string
      dependencies: cuda
    cuda_requires:
      type: string
      dependencies: cuda
    needs_deps_image:
      type: boolean
    envars:
      type: list
      default: []
      schema:
        check_with: args_format
        type: string
    cuda:
      type: dict

releases:
  type: dict
  keysrules:
    supported_dists: true
    type: string
  valuesrules:
    type: dict
    keysrules:
      regex: '([\d\.]+)'
    valuesrules:
      type: dict
    
"""


def validate_template_name(names):
    """Validate our input file name for Dockerfile templates."""
    if isinstance(names, str):
        if REPO_PREFIX in names:
            # ignore .repo.j2 for centos
            pass
        elif DOCKERFILE_PREFIX in names:
            names, _, _ = names.partition(DOCKERFILE_PREFIX)
            if names not in DOCKERFILE_ALLOWED_NAMES:
                raise RuntimeError(
                    f"Unable to parse {names}. Dockerfile directory should consists of {', '.join(DOCKERFILE_ALLOWED_NAMES)}"
                )
        else:
            raise RuntimeWarning(
                f"{inspect.currentframe().f_code.co_name} is being used to validate {names}, which has unknown extensions, "
                f"thus will have no effects. "
            )
            pass
    elif isinstance(names, list):
        for _name in names:
            validate_template_name(_name)
    else:
        raise TypeError(f"{names} has unknown type {type(names)}")


def get_data(obj, *path, skip=False):
    """
    Get data from the object by dotted path.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        data = glom(obj, Path(*path))
    except PathAccessError:
        if skip:
            return
        logging.error(
            f"Exception occurred in get_data, unable to retrieve from {' '.join(*path)}"
        )
    else:
        return data


def update_data(obj, value, *path):
    """
    Update data from the object with given value.
    e.g: dependencies.cuda."11.3.1"
    """
    try:
        _ = glom(obj, Assign(Path(*path), value))
    except PathAssignError:
        logging.error(
            f"Exception occurred in update_data, unable to update {Path(*path)} with {value}"
        )


def _build_release_args(bentoml_version):
    """
    Create a dictionary of args that will be parsed during runtime

    Args:
        bentoml_version: version of BentoML releases

    Returns:
        Dict of args
    """
    # we will update PYTHON_VERSION when build process is invoked
    args = {"PYTHON_VERSION": "", "BENTOML_VERSION": bentoml_version}

    def update_args_dict(updater):
        # updater will be distro_release_spec['envars'] (e.g)
        # Args will have format ARG=foobar
        if isinstance(updater, str):
            _args, _, _value = updater.partition("=")
            args[_args] = _value
            return
        elif isinstance(updater, list):
            for v in updater:
                update_args_dict(v)
        elif isinstance(updater, dict):
            for _args, _value in updater.items():
                args[_args] = _value
        else:
            raise AttributeError(
                f"cannot add to args dict with unknown type {type(updater)}"
            )
        return args

    return update_args_dict


def _build_release_tags(build_ctx, release_type):
    """
    Our tags will have format: {bentoml_release_version}-{python_version}-{distros}-{image_type}
    eg:
    - python3.8-ubuntu20.04-base -> contains conda and python preinstalled
    - devel-python3.8-ubuntu20.04
    - 0.13.0-python3.7-centos8-{runtime,cudnn}
    """

    _python_version = get_data(build_ctx, 'envars', 'PYTHON_VERSION')

    # check if PYTHON_VERSION is set.
    if not _python_version:
        raise RuntimeError("PYTHON_VERSION should be set before generating Dockerfile.")
    if _python_version not in SUPPORTED_PYTHON_VERSION:
        raise ValueError(
            f"{_python_version} is not supported. Supported versions: {SUPPORTED_PYTHON_VERSION}"
        )

    _tag = ""
    core_string = '-'.join([f"python{_python_version}", build_ctx['add_to_tags']])

    base_tag = '-'.join([core_string, 'base'])

    if release_type == "devel":
        _tag = "-".join([release_type, core_string])
    elif release_type == "base":
        pass
    else:
        _tag = "-".join(
            [
                get_data(build_ctx, 'envars', 'BENTOML_VERSION'),
                core_string,
                release_type,
            ]
        )
    return _tag, base_tag


def flatten(arr):
    for it in arr:
        if isinstance(it, Iterable) and not isinstance(it, str):
            for x in flatten(it):
                yield x
        else:
            yield it


def load_manifest_yaml(file):
    with open(file, 'r') as input_file:
        manifest = yaml.safe_load(input_file)

    v_schema = yaml.safe_load(SPEC_SCHEMA)
    v = MetadataSpecValidator(
        v_schema, packages=manifest['packages'], releases=manifest['releases']
    )

    if not v.validate(manifest):
        logging.error(f"{file} is invalid. Errors as follow:\n")
        logging.error(yaml.dump(v.errors, indent=2))
        exit(1)

    release_spec = v.normalized(manifest)
    return release_spec


def _update_build_ctx(ctx_dict, spec_dict: Dict, *ignore_keys):
    # flatten release_spec to template context.
    for k, v in spec_dict.items():
        if k in list(flatten([*ignore_keys])):
            continue
        ctx_dict[k] = v


def _add_base_release_type(build_ctx, release_type: list):
    try:
        return release_type + [build_ctx['base_tags']]
    except KeyError as e:
        # when base_tags is either not set or available
        return release_type


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

    # TODO: custom check for releases.valuesrules

    def __init__(self, *args, **kwargs):
        if 'packages' in kwargs:
            self.packages = kwargs['packages']
        if 'releases' in kwargs:
            self.releases = kwargs['releases']
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

    def _check_with_dists_defined(self, field, value):
        if isinstance(value, list):
            for v in value:
                self._check_with_dists_defined(field, v)
        else:
            if value not in self.releases:
                self._error(field, f"{field} is not defined under releases")

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
                    f"{value} is not defined in SUPPORTED_OS. If you are adding a new distros make "
                    f"sure to add it to SUPPORTED_OS",
                )
        if isinstance(value, list):
            for v in value:
                self._validate_supported_dists(supported_dists, field, v)


# noinspection PyPep8Naming
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
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
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


class TemplatesMixin(object):
    """Mixin for generating Dockerfile from templates."""

    _build_ctx = {}
    # _build_path will consist of
    #   key: package_tag
    #   value: path to dockerfile
    _build_path = {}

    _template_env = Environment(
        extensions=['jinja2.ext.do'], trim_blocks=True, lstrip_blocks=True
    )

    def __init__(self, release_spec, cuda_version):
        # set private class attributes to be manifest.yml keys
        for keys, values in release_spec.items():
            super().__setattr__(f"_{keys}", values)

        # setup default keys for build context
        _update_build_ctx(self._build_ctx, self._release_spec, 'cuda')

        # setup cuda_version and cuda_components
        if self._check_cuda_version(cuda_version):
            self._cuda_version = cuda_version
        else:
            raise RuntimeError(
                f"CUDA {cuda_version} is either not supported or not defined in {self._cuda_dependencies.keys()}"
            )
        self._cuda_components = get_data(self._cuda_dependencies, self._cuda_version)

    def __call__(self, dry_run=False, *args, **kwargs):
        if not dry_run:
            if FLAGS.python_version is not None:
                logging.info(
                    "--python_version defined, ignoring SUPPORTED_PYTHON_VERSION"
                )
                supported_python = [FLAGS.python_version]
            else:
                supported_python = SUPPORTED_PYTHON_VERSION

            # we will remove generated dockerfile dir everytime generating new BentoML release.
            logging.info(f"Removing dockerfile dir: {FLAGS.dockerfile_dir}")
            shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
            try:
                os.makedirs(FLAGS.dockerfile_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            self.generate_dockerfiles(
                target_dir=FLAGS.dockerfile_dir, python_version=supported_python
            )

    @cached_property
    def build_path(self):
        return self._build_path

    def _cudnn_versions(self):
        for k, v in self._cuda_components.items():
            if k.startswith('cudnn') and v:
                return k, v

    def _check_cuda_version(self, cuda_version):
        # get defined cuda version for cuda
        return True if cuda_version in self._cuda_dependencies.keys() else False

    def _generate_distro_release_mapping(self, package):
        """Return a dict containing keys as distro and values as release type (devel, cudnn, runtime)"""
        releases = defaultdict(list)
        for k, v in self._packages[package].items():
            _v = list(flatten(v))
            for distro in _v:
                releases[distro].append(k)
        return releases

    def prepare_template_context(
        self, distro, distro_version, distro_release_spec, python_version
    ):
        """
        Generate template context for each distro releases.

        build context structure:
            {
              "templates_dir": "",
              "base_image": "",
              "add_to_tags": "",
              "envars": {},
              "cuda_prefix_url": "",
              "cuda": {
                "ml_repo": "",
                "base_repo": "",
                "version": {},
                "components": {}
              }
            }
        """
        _build_ctx = deepcopy(self._build_ctx)

        cuda_regex = re.compile(r"cuda*")
        cudnn_regex = re.compile(r"(cudnn)(\w+)")

        _update_build_ctx(
            _build_ctx, get_data(self._releases, distro, distro_version), 'cuda'
        )

        _build_ctx['envars'] = _build_release_args(FLAGS.bentoml_version)(
            distro_release_spec['envars']
        )

        # setup base_tags if needed
        if _build_ctx['needs_deps_image'] is True:
            _build_ctx['base_tags'] = 'base'
        _build_ctx.pop('needs_deps_image')

        # setup python version
        update_data(
            _build_ctx, python_version, "envars", "PYTHON_VERSION",
        )

        # setup cuda deps
        if _build_ctx['cuda_prefix_url']:
            major, minor, _ = self._cuda_version.rsplit(".")
            cudnn_version_name, cudnn_release_version = self._cudnn_versions()
            cudnn_groups = cudnn_regex.match(cudnn_version_name).groups()
            cudnn_namespace, cudnn_major_version = [
                cudnn_groups[i] for i in range(len(cudnn_groups))
            ]
            _build_ctx['cuda'] = {
                'ml_repo': NVIDIA_ML_REPO_URL.format(_build_ctx['cuda_prefix_url']),
                'base_repo': NVIDIA_REPO_URL.format(_build_ctx['cuda_prefix_url']),
                'version': {
                    'release': self._cuda_version,
                    'major': major,
                    'minor': minor,
                    'shortened': ".".join([major, minor]),
                },
                cudnn_namespace: {
                    'version': cudnn_release_version,
                    'major_version': cudnn_major_version,
                },
                'components': self._cuda_components,
            }
            _build_ctx.pop('cuda_prefix_url')
        else:
            logging.debug("distro doesn't enable CUDA. Skipping...")
            for key in list(_build_ctx.keys()):
                if cuda_regex.match(key):
                    _build_ctx.pop(key)

        return _build_ctx

    def render_dockerfile_from_templates(
        self,
        input_tmpl_path: pathlib.Path,
        output_path: pathlib.Path,
        ctx: Dict,
        tags: Optional[Dict[str, str]] = None,
    ):
        with open(input_tmpl_path, 'r') as inf:
            logging.debug(f"Processing template: {input_tmpl_path}")
            name = input_tmpl_path.name
            if DOCKERFILE_PREFIX not in name:
                logging.debug(f"Processing {name}")
                output_name = name[: -len('.j2')]
            else:
                output_name = DOCKERFILE_PREFIX.split('.')[1]

            template = self._template_env.from_string(inf.read())
            if not output_path.exists():
                logging.debug(f"Creating {output_path}")
                output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_path}/{output_name}", 'w') as ouf:
                ouf.write(template.render(metadata=ctx, tags=tags))
            ouf.close()
        inf.close()

    def generate_dockerfiles(self, target_dir, python_version):
        """
        Workflow:
        walk all templates dir from given context

        our targeted generate directory should look like:
        generated
            -- packages (model-server, yatai-service)
                -- python_version (3.8,3.7,etc)
                    -- {distro}{distro_version} (ubuntu20.04, ubuntu18.04, centos8, centos7, amazonlinux2, etc)
                        -- runtime
                            -- Dockerfile
                        -- devel
                            -- Dockerfile
                        -- cudnn
                            -- Dockerfile
                        -- Dockerfile (for base deps images)

        Args:
            target_dir: parent directory of generated Dockerfile. Default: ./generated.
            python_version: target list of supported python version. Overwrite default supports list if
                            FLAGS.python_version is given.

        Returns:
            Let the magic happens.
        """
        _tag_metadata = defaultdict(list)

        logging.info(f"Generating dockerfile to {target_dir}...\n")
        for package in self._packages.keys():
            # update our packages ctx.
            logging.info(f"> Working on {package}")
            update_data(self._build_ctx, package, "package")
            for _py_ver in python_version:
                logging.info(f">> Using python{_py_ver}")
                for distro, _release_type in self._generate_distro_release_mapping(
                    package
                ).items():
                    for distro_version, distro_spec in self._releases[distro].items():
                        # setup our build context
                        _build_ctx = self.prepare_template_context(
                            distro, distro_version, distro_spec, _py_ver
                        )

                        logging.info(f">>> Generating: {distro}{distro_version}")

                        # validate our template directory.
                        for _, _, files in os.walk(_build_ctx['templates_dir']):
                            validate_template_name(files)

                        for _re_type in _add_base_release_type(
                            _build_ctx, _release_type
                        ):
                            # setup filepath
                            input_tmpl_path = pathlib.Path(
                                _build_ctx['templates_dir'],
                                f"{_re_type}{DOCKERFILE_PREFIX}",
                            )

                            base_output_path = pathlib.Path(
                                target_dir,
                                package,
                                _py_ver,
                                f'{distro}{distro_version}',
                            )

                            # handle cuda.repo and nvidia-ml.repo
                            if (
                                'rhel' in _build_ctx['templates_dir']
                                and _re_type == "cudnn"
                            ):
                                for _n in ['nvidia-ml', 'cuda']:
                                    repo_tmpl_path = pathlib.Path(
                                        _build_ctx['templates_dir'],
                                        f"{_n}{REPO_PREFIX}",
                                    )
                                    print(repo_tmpl_path)
                                    repo_output_path = pathlib.Path(
                                        base_output_path, _re_type
                                    )
                                    self.render_dockerfile_from_templates(
                                        repo_tmpl_path, repo_output_path, _build_ctx
                                    )

                            # generate our image tag.
                            tag, basetag = _build_release_tags(_build_ctx, _re_type)
                            tag_ctx = {"basename": basetag}

                            if _re_type != "base":
                                output_path = pathlib.Path(base_output_path, _re_type)
                            else:
                                output_path = base_output_path

                            # setup directory correspondingly, and add these paths
                            # with correct tags name under self._build_path
                            if tag:
                                tag_keys = f"{_build_ctx['package']}:{tag}"
                            else:
                                tag_keys = f"{_build_ctx['package']}:{basetag}"

                            _tag_metadata[tag_keys].append(_build_ctx)
                            self._build_path[tag_keys] = str(
                                pathlib.Path(output_path, 'Dockerfile')
                            )
                            # generate our Dockerfile from templates.
                            self.render_dockerfile_from_templates(
                                input_tmpl_path=input_tmpl_path,
                                output_path=output_path,
                                ctx=_build_ctx,
                                tags=tag_ctx,
                            )

                logging.info(
                    f">> target directory for python{_py_ver}: {target_dir}/{package}/{_py_ver}\n"
                )
            logging.info(">" * 50 + "\n")

        if FLAGS.dump_metadata:
            file = "./metadata.json"
            logging.info(f"> Dumping metadata to {file}")
            with open(file, "w") as ouf:
                ouf.write(json.dumps(_tag_metadata, indent=2))
            ouf.close()
            del _tag_metadata


def main(argv):
    if len(argv) > 1:
        raise RuntimeError("Too much arguments")

    # Parse specs from manifest.yml and validate it.
    release_spec = load_manifest_yaml(FLAGS.manifest_file)

    # generate templates to correct directory.
    tmpl_mixin = TemplatesMixin(release_spec, cuda_version=FLAGS.cuda_version)
    tmpl_mixin(dry_run=FLAGS.dry_run)

    if FLAGS.stop_at_generate:
        logging.info("--stop_at_generate is parsed. Stopping now...")
        return

    t = PrettyTable(["image_tag", "dockerfile_path"])
    for k, v in tmpl_mixin.build_path.items():
        t.add_row([k, v])
    print(t)

    # Setup Docker credentials
    # this includes push directory, envars


if __name__ == "__main__":
    app.run(main)