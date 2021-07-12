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

import logging
import os
from pathlib import Path
import stat
from sys import version_info
from typing import List

from simple_di import Provide, inject

from bentoml.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException
from bentoml.saved_bundle.pip_pkg import (
    EPP_NO_ERROR,
    EPP_PKG_NOT_EXIST,
    EPP_PKG_VERSION_MISMATCH,
    get_pkg_version,
    verify_pkg,
)
from bentoml.utils.ruamel_yaml import YAML
from bentoml.utils import cached_property


logger = logging.getLogger(__name__)

PYTHON_SUPPORTED_VERSIONS = ["3.6", "3.7", "3.8"]

PYTHON_MINOR_VERSION = "{major}.{minor}".format(
    major=version_info.major, minor=version_info.minor
)

PYTHON_VERSION = "{minor_version}.{micro}".format(
    minor_version=PYTHON_MINOR_VERSION, micro=version_info.micro
)


DEFAULT_CONDA_ENV_BASE_YAML = """
name: bentoml-default-conda-env
dependencies: []
"""


class CondaEnv(object):
    """A wrapper around conda environment settings file, allows adding/removing
    conda or pip dependencies to env configuration, and supports load/export those
    settings from/to yaml files. The generated file is the same format as yaml file
    generated from `conda env export` command.
    """

    def __init__(
        self,
        name: str = None,
        channels: List[str] = None,
        dependencies: List[str] = None,
        default_env_yaml_file: str = None,
        override_channels: bool = False,
    ):
        self._yaml = YAML()
        self._yaml.default_flow_style = False
        self.default_env_yaml_file = default_env_yaml_file

        if name:
            self.set_name(name)

        if override_channels and channels is None:
            raise BentoMLException(
                "No `conda_channels` provided while override_channels=True"
            )

        if channels:
            self.set_channels(channels, override_channels)

        if dependencies:
            self.add_conda_dependencies(dependencies)

    @cached_property
    def _conda_env(self):
        if self.default_env_yaml_file:
            env_yml_file = Path(self.default_env_yaml_file)
            if not env_yml_file.is_file():
                raise BentoMLException(
                    f"Can not find conda environment config yaml file at: "
                    f"`{self.default_env_yaml_file}`"
                )
            return self._yaml.load(env_yml_file)
        else:
            return self._yaml.load(DEFAULT_CONDA_ENV_BASE_YAML)

    def set_name(self, name):
        self._conda_env["name"] = name

    def add_conda_dependencies(self, conda_dependencies: List[str]):
        # BentoML uses conda's channel_priority=strict option by default
        # Adding `dependencies` to beginning of the list to take priority over the
        # existing conda channels
        self._conda_env["dependencies"] = (
            conda_dependencies + self._conda_env["dependencies"]
        )

    def set_channels(self, channels: List[str], override_channels=False):
        if override_channels and "nodefaults" not in channels:
            channels.append("nodefaults")
        self._conda_env["channels"] = channels

    def write_to_yaml_file(self, filepath):
        with open(filepath, 'wb') as output_yaml:
            self._yaml.dump(self._conda_env, output_yaml)


class BentoServiceEnv(object):
    """Defines all aspect of the system environment requirements for a custom
    BentoService to be used. This includes:


    Args:
        pip_packages: list of pip_packages required, specified by package name
            or with specified version `{package_name}=={package_version}`
        pip_index_url: passing down to pip install --index-url option
        pip_trusted_host: passing down to pip install --trusted-host option
        pip_extra_index_url: passing down to pip install --extra-index-url option
        infer_pip_packages: Turn on to automatically find all the required
            pip dependencies and pin their version
        requirements_txt_file: path to the requirements.txt where pip dependencies
            are explicitly specified, with ideally pinned versions
        conda_channels: list of extra conda channels other than default channels to be
            used. This is equivalent to passing the --channels to conda commands.
            If the `conda_env_yml_file` is specified, this will override the `channels`
            section of the env yml file
        conda_override_channels: ensures that conda searches only your specified
            channel and no other channels, such as default channels.
            This is equivalent to passing the --override-channels option to conda
            commands, or adding `nodefaults` to the `channels` in the environment.yml
        conda_overwrite_channels: aliases to `override_channels`
        conda_dependencies: list of conda dependencies required
        conda_env_yml_file: use a pre-defined conda environment yml file
        setup_sh: user defined setup bash script, it is executed in docker build time
        docker_base_image: used when generating Dockerfile in saved bundle
        zipimport_archives: used to list zipimport archives
    """

    @inject
    def __init__(
        self,
        pip_packages: List[str] = None,
        pip_index_url: str = None,
        pip_trusted_host: str = None,
        pip_extra_index_url: str = None,
        infer_pip_packages: bool = False,
        requirements_txt_file: str = None,
        conda_channels: List[str] = None,
        conda_overwrite_channels: bool = False,
        conda_override_channels: bool = False,
        conda_dependencies: List[str] = None,
        conda_env_yml_file: str = None,
        setup_sh: str = None,
        docker_base_image: str = None,
        default_docker_base_image: str = Provide[
            BentoMLContainer.config.bento_bundle.default_docker_base_image
        ],
        zipimport_archives: List[str] = None,
        bentoml_deployment_version: str = Provide[
            BentoMLContainer.bento_bundle_deployment_version
        ],
    ):
        self._python_version = PYTHON_VERSION
        self._pip_index_url = pip_index_url
        self._pip_trusted_host = pip_trusted_host
        self._pip_extra_index_url = pip_extra_index_url
        self._requirements_txt_file = requirements_txt_file
        self._requirements_txt_content = None

        self._conda_env = CondaEnv(
            channels=conda_channels,
            dependencies=conda_dependencies,
            default_env_yaml_file=conda_env_yml_file,
            override_channels=conda_overwrite_channels or conda_override_channels,
        )

        self._pip_packages = {}

        # add BentoML to pip packages list
        bentoml_deploy_version = bentoml_deployment_version
        self.add_pip_package("bentoml=={}".format(bentoml_deploy_version))

        if pip_packages:
            self.add_pip_packages(pip_packages)

        self._infer_pip_packages = infer_pip_packages

        self.set_setup_sh(setup_sh)

        if docker_base_image:
            logger.info(
                f"Using user specified docker base image: `{docker_base_image}`, user"
                f"must make sure that the base image either has Python "
                f"{PYTHON_MINOR_VERSION} or conda installed."
            )
            self._docker_base_image = docker_base_image
        elif default_docker_base_image:
            logger.info(
                f"Using default docker base image: `{docker_base_image}` specified in"
                f"BentoML config file or env var. User must make sure that the docker "
                f"base image either has Python {PYTHON_MINOR_VERSION} or conda "
                f"installed."
            )
            self._docker_base_image = default_docker_base_image
        else:
            if PYTHON_MINOR_VERSION not in PYTHON_SUPPORTED_VERSIONS:
                self._docker_base_image = (
                    f"bentoml/model-server:{bentoml_deploy_version}"
                )

                logger.warning(
                    f"Python {PYTHON_VERSION} found in current environment is not "
                    f"officially supported by BentoML. The docker base image used is"
                    f"'{self._docker_base_image}' which will use conda to install "
                    f"Python {PYTHON_VERSION} in the build process. Supported Python "
                    f"versions are: f{', '.join(PYTHON_SUPPORTED_VERSIONS)}"
                )
            else:
                # e.g. bentoml/model-server:0.8.6-py37
                self._docker_base_image = (
                    f"bentoml/model-server:"
                    f"{bentoml_deploy_version}-"
                    f"py{PYTHON_MINOR_VERSION.replace('.', '')}"
                )

                logger.debug(
                    f"Using BentoML default docker base image "
                    f"'{self._docker_base_image}'"
                )
        self._zipimport_archives = zipimport_archives

    def add_conda_dependencies(self, conda_dependencies: List[str]):
        self._conda_env.add_conda_dependencies(conda_dependencies)

    def add_pip_packages(self, pip_packages: List[str]):
        for pip_package in pip_packages:
            self.add_pip_package(pip_package)

    def add_pip_package(self, pip_package: str):
        # str must be a valid pip requirement specifier
        # https://pip.pypa.io/en/stable/reference/pip_install/#requirement-specifiers
        from pkg_resources import Requirement

        package_req = Requirement(pip_package)
        self._add_pip_package_requirement(package_req)

    def _add_pip_package_requirement(self, pkg_req):
        from pkg_resources import Requirement

        if pkg_req.name in self._pip_packages:
            if (
                pkg_req.specs
                and pkg_req.specs != self._pip_packages[pkg_req.name].specs
            ):
                logger.warning(
                    f"Overwriting existing pip package requirement "
                    f"'{self._pip_packages[pkg_req.name]}' to '{pkg_req}'"
                )
            else:
                logger.debug(f"pip package requirement '{pkg_req}' already exist")
                return

        verification_result = verify_pkg(pkg_req)
        if verification_result == EPP_PKG_NOT_EXIST:
            logger.warning(
                f"pip package requirement `{str(pkg_req)}` not found in current "
                f"python environment"
            )
        elif verification_result == EPP_PKG_VERSION_MISMATCH:
            logger.warning(
                f"pip package requirement `{str(pkg_req)}` does not match the "
                f"version installed in current python environment"
            )

        if verification_result == EPP_NO_ERROR and not pkg_req.specs:
            # pin the current version when there's no version spec found
            pkg_version = get_pkg_version(pkg_req.name)
            if pkg_version:
                pkg_req = Requirement(f"{pkg_req.name}=={pkg_version}")

        self._pip_packages[pkg_req.name] = pkg_req

    def set_setup_sh(self, setup_sh_path_or_content):
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

    def requirements_txt_content(self):
        if self._requirements_txt_content:
            # This branch handles the case when the _requirements_txt_content is
            # previously set by the BentoBundle loader
            return self._requirements_txt_content

        if not self._requirements_txt_file:
            raise BentoMLException("requirement txt file not specified")

        if not self._requirements_txt_content:
            req_txt_file = Path(self._requirements_txt_file)
            if not req_txt_file.is_file():
                raise BentoMLException(
                    f"requirement txt file not found at '{self._requirements_txt_file}'"
                )
            self._requirements_txt_content = req_txt_file.read_text()

        return self._requirements_txt_content

    def infer_pip_packages(self, bento_service):
        if self._infer_pip_packages:
            dependencies_map = bento_service.infer_pip_dependencies_map()
            for pkg_name, pkg_version in dependencies_map.items():
                if (
                    pkg_name in self._pip_packages
                    and self._pip_packages[pkg_name].specs
                ):
                    # package exist and already have version specs associated
                    continue
                else:
                    # pin current version if the package has not been added
                    self.add_pip_package(f"{pkg_name}=={pkg_version}")

    def save(self, path):
        conda_yml_file = os.path.join(path, "environment.yml")
        self._conda_env.write_to_yaml_file(conda_yml_file)

        with open(os.path.join(path, "python_version"), "wb") as f:
            f.write(self._python_version.encode("utf-8"))

        # If requirements.txt is specified, other requirements will
        # NOT be parsed (#1421)
        requirements_txt_file = os.path.join(path, "requirements.txt")
        with open(requirements_txt_file, "wb") as f:
            if self._requirements_txt_file:
                f.write(self.requirements_txt_content().encode("utf-8"))
            else:
                if self._pip_index_url:
                    f.write(f"--index-url={self._pip_index_url}\n".encode("utf-8"))
                if self._pip_trusted_host:
                    f.write(
                        f"--trusted-host={self._pip_trusted_host}\n".encode("utf-8")
                    )
                if self._pip_extra_index_url:
                    f.write(
                        f"--extra-index-url={self._pip_extra_index_url}\n".encode(
                            "utf-8"
                        )
                    )
                pip_content = '\n'.join(
                    [str(pkg_req) for pkg_req in self._pip_packages.values()]
                ).encode("utf-8")
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

        if self._requirements_txt_file:
            env_dict["requirements_txt"] = self.requirements_txt_content()
        elif self._pip_packages:
            # pip_packages are ignored when the requirements_txt_file parameter is
            # specified by the user
            env_dict["pip_packages"] = [
                str(pkg_req) for pkg_req in self._pip_packages.values()
            ]

        env_dict["conda_env"] = self._conda_env._conda_env
        env_dict["python_version"] = self._python_version
        env_dict["docker_base_image"] = self._docker_base_image
        return env_dict
