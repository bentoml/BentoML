# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
from sys import version_info
import stat
from pathlib import Path

from ruamel.yaml import YAML

from bentoml import config
from bentoml.configuration import get_bentoml_deploy_version
from bentoml.utils.pip_pkg import (
    EPP_PKG_NOT_EXIST,
    EPP_PKG_VERSION_MISMATCH,
    parse_requirement_string,
    verify_pkg,
    seek_pip_dependencies,
)


logger = logging.getLogger(__name__)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)

CONDA_ENV_BASE_YAML = """
name: {name}
channels:
  - defaults
dependencies:
  - python={python_version}
  - pip
"""

CONDA_ENV_DEFAULT_NAME = "bentoml-custom-conda-env"


class CondaEnv(object):
    """A wrapper around conda environment settings file, allows adding/removing
    conda or pip dependencies to env configuration, and supports load/export those
    settings from/to yaml files. The generated file is the same format as yaml file
    generated from `conda env export` command.
    """

    def __init__(self, name, python_version):
        self._yaml = YAML()
        self._yaml.default_flow_style = False

        self._conda_env = self._yaml.load(
            CONDA_ENV_BASE_YAML.format(name=name, python_version=python_version)
        )

    def set_name(self, name):
        self._conda_env["name"] = name

    def get_name(self):
        return self._conda_env["name"]

    def add_conda_dependencies(self, extra_conda_dependencies):
        self._conda_env["dependencies"] += extra_conda_dependencies

    def add_channels(self, channels):
        self._conda_env["channels"] += channels

    def write_to_yaml_file(self, filepath):
        with open(filepath, 'wb') as output_yaml:
            self._yaml.dump(self._conda_env, output_yaml)


class BentoServiceEnv(object):
    """Defines all aspect of the system environment requirements for a custom
    BentoService to be used. This includes:


    Args:
        bento_service_name: name of the BentoService name bundled with this Env
        pip_dependencies: list of pip_dependencies required, specified by package name
            or with specified version `{package_name}=={package_version}`
        auto_pip_dependencies: (Beta) whether to automatically find all the required
            pip dependencies and pin their version
        requirements_txt_file: pip dependencies in the form of a requirements.txt file,
            this can be a relative path to the requirements.txt file or the content
            of the file
        conda_channels: extra conda channels to be used
        conda_dependencies: list of conda dependencies required
        setup_sh: user defined setup bash script, it is executed in docker build time
        docker_base_image: used when generating Dockerfile in saved bundle
    """

    def __init__(
        self,
        bento_service_name,
        pip_dependencies=None,
        auto_pip_dependencies=False,
        requirements_txt_file=None,
        conda_channels=None,
        conda_dependencies=None,
        setup_sh=None,
        docker_base_image=None,
    ):
        self._python_version = PYTHON_VERSION

        self._conda_env = CondaEnv(
            "bentoml-" + bento_service_name, self._python_version
        )

        bentoml_deploy_version = get_bentoml_deploy_version()
        self._pip_dependencies = ["bentoml=={}".format(bentoml_deploy_version)]
        if pip_dependencies:
            if auto_pip_dependencies:
                logger.warning(
                    "auto_pip_dependencies enabled, it may override package versions "
                    "specified in `pip_dependencies=%s`",
                    pip_dependencies,
                )
            else:
                for dependency in pip_dependencies:
                    self.check_dependency(dependency)
                self._pip_dependencies += pip_dependencies

        if requirements_txt_file:
            if auto_pip_dependencies:
                logger.warning(
                    "auto_pip_dependencies enabled, it may override package versions "
                    "specified in `requirements_txt_file=%s`",
                    requirements_txt_file,
                )
            else:
                self._set_requirements_txt(requirements_txt_file)

        self._auto_pip_dependencies = auto_pip_dependencies

        self._set_setup_sh(setup_sh)

        if conda_channels:
            self._add_conda_channels(conda_channels)
        if conda_dependencies:
            self._add_conda_dependencies(conda_dependencies)

        if docker_base_image:
            self._docker_base_image = docker_base_image
        else:
            self._docker_base_image = config('core').get('default_docker_base_image')

    @staticmethod
    def check_dependency(dependency):
        name, version = parse_requirement_string(dependency)
        code = verify_pkg(name, version)
        if code == EPP_PKG_NOT_EXIST:
            logger.warning(
                '%s package does not exist in the current python ' 'session', name
            )
        elif code == EPP_PKG_VERSION_MISMATCH:
            logger.warning(
                '%s package version is different from the version '
                'being used in the current python session',
                name,
            )

    def get_conda_env_name(self):
        return self._conda_env.get_name()

    def set_conda_env_name(self, name):
        self._conda_env.set_name(name)

    def _add_conda_channels(self, channels):
        if not isinstance(channels, list):
            channels = [channels]
        self._conda_env.add_channels(channels)

    def _add_conda_dependencies(self, conda_dependencies):
        if not isinstance(conda_dependencies, list):
            conda_dependencies = [conda_dependencies]
        self._conda_env.add_conda_dependencies(conda_dependencies)

    def _add_pip_dependencies(self, pip_dependencies):
        if not isinstance(pip_dependencies, list):
            pip_dependencies = [pip_dependencies]
        self._pip_dependencies += pip_dependencies

    def _add_pip_dependencies_if_missing(self, pip_dependencies):
        # Insert dependencies to the beginning of self.dependencies, so that user
        # specified dependency version could overwrite this. This is used by BentoML
        # to inject ModelArtifact or Handler's optional pip dependencies
        if not isinstance(pip_dependencies, list):
            pip_dependencies.insert(0, pip_dependencies)
        self._pip_dependencies = pip_dependencies + self._pip_dependencies

    def _set_setup_sh(self, setup_sh_path_or_content):
        self._setup_sh = None

        if setup_sh_path_or_content:
            setup_sh_file = Path(setup_sh_path_or_content)
        else:
            return

        if setup_sh_file.is_file():
            with setup_sh_file.open("rb") as f:
                self._setup_sh = f.read()
        else:
            self._setup_sh = setup_sh_path_or_content.encode("utf-8")

    def _set_requirements_txt(self, requirements_txt_path):
        requirements_txt_file = Path(requirements_txt_path)

        with requirements_txt_file.open("rb") as f:
            content = f.read()
            module_list = content.decode("utf-8").split("\n")
            self._pip_dependencies += module_list

    def save(self, path, bento_service):
        conda_yml_file = os.path.join(path, "environment.yml")
        self._conda_env.write_to_yaml_file(conda_yml_file)

        requirements_txt_file = os.path.join(path, "requirements.txt")

        with open(requirements_txt_file, "wb") as f:
            dependencies_map = {}
            for dep in self._pip_dependencies:
                name, version = parse_requirement_string(dep)
                dependencies_map[name] = version

            if self._auto_pip_dependencies:
                bento_service_module = sys.modules[bento_service.__class__.__module__]
                if hasattr(bento_service_module, "__file__"):
                    bento_service_py_file_path = bento_service_module.__file__
                    reqs, unknown_modules = seek_pip_dependencies(
                        bento_service_py_file_path
                    )
                    dependencies_map.update(reqs)
                    for module_name in unknown_modules:
                        logger.warning(
                            "unknown package dependency for module: %s", module_name
                        )

                # Reset bentoml to configured deploy version - this is for users using
                # customized BentoML branch for development but use a different stable
                # version for deployment
                #
                # For example, a BentoService created with local dirty branch will fail
                # to deploy with docker due to the version can't be found on PyPI, but
                # get_bentoml_deploy_version gives the user the latest released PyPI
                # version that's closest to the `dirty` branch
                dependencies_map['bentoml'] = get_bentoml_deploy_version()

            # Set self._pip_dependencies so it get writes to BentoService config file
            self._pip_dependencies = []
            for pkg_name, pkg_version in dependencies_map.items():
                self._pip_dependencies.append(
                    "{}{}".format(
                        pkg_name, "=={}".format(pkg_version) if pkg_version else ""
                    )
                )

            pip_content = "\n".join(self._pip_dependencies).encode("utf-8")
            f.write(pip_content)

        if self._setup_sh:
            setup_sh_file = os.path.join(path, "setup.sh")
            with open(setup_sh_file, "wb") as f:
                f.write(self._setup_sh)

            # chmod +x setup.sh
            st = os.stat(setup_sh_file)
            os.chmod(setup_sh_file, st.st_mode | stat.S_IEXEC)

    def to_dict(self):
        env_dict = dict()

        if self._setup_sh:
            env_dict["setup_sh"] = self._setup_sh

        if self._pip_dependencies:
            env_dict["pip_dependencies"] = self._pip_dependencies

        env_dict["conda_env"] = self._conda_env._conda_env

        env_dict["python_version"] = self._python_version

        env_dict["docker_base_image"] = self._docker_base_image
        return env_dict
