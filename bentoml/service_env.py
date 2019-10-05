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
from sys import version_info
from ruamel.yaml import YAML

from bentoml.utils import Path, StringIO
from bentoml.configuration import get_bentoml_deploy_version

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
  - pip:
    - bentoml[api_server]=={bentoml_version}
"""

CONDA_ENV_DEFAULT_NAME = "bentoml-custom-conda-env"


class CondaEnv(object):
    """A wrapper around conda environment settings file, allows adding/removing
    conda or pip dependencies to env configuration, and supports load/export those
    settings from/to yaml files. The generated file is the same format as yaml file
    generated from `conda env export` command.
    """

    def __init__(
        self,
        name=CONDA_ENV_DEFAULT_NAME,
        python_version=PYTHON_VERSION,
        bentoml_version=None,
    ):
        self._yaml = YAML()
        self._yaml.default_flow_style = False

        if bentoml_version is None:
            bentoml_version = get_bentoml_deploy_version()

        self._conda_env = self._yaml.load(
            CONDA_ENV_BASE_YAML.format(
                name=name,
                python_version=python_version,
                bentoml_version=bentoml_version,
            )
        )

    def set_name(self, name):
        self._conda_env["name"] = name

    def get_name(self):
        return self._conda_env["name"]

    def add_conda_dependencies(self, extra_conda_dependencies):
        self._conda_env["dependencies"] += extra_conda_dependencies

    def add_pip_dependencies(self, extra_pip_dependencies):
        for dep in self._conda_env["dependencies"]:
            if isinstance(dep, dict) and "pip" in dep:
                # there is already a pip list in conda_env, append extra deps
                dep["pip"] += extra_pip_dependencies
                return self

        self._conda_env["dependencies"] += [{"pip": extra_pip_dependencies}]

    def add_channels(self, channels):
        self._conda_env["channels"] += channels

    def to_yaml_str(self):
        string_io = StringIO()
        self._yaml.dump(self._conda_env, string_io)
        return string_io.getvalue()

    def write_to_yaml_file(self, filepath):
        output_yaml = Path(filepath)
        self._yaml.dump(self._conda_env, output_yaml)

    @classmethod
    def from_yaml(cls, yaml_str):
        new_conda_env = cls()
        new_conda_env._conda_env = new_conda_env._yaml.load(yaml_str)
        return new_conda_env

    @classmethod
    def from_current_conda_env(cls):
        # TODO: implement me!
        pass


class BentoServiceEnv(object):
    """Defines all aspect of the system environment requirements for a custom
    BentoService to be used. This includes:

    conda environment - for most python third-party packages and libraries
    requirements_txt  - for pypi dependencies that can be resolved by pip
        when exported BentoArchieve is installed as a pypi package
    setup_sh - for customizing the environment with user defined bash script
    """

    def __init__(self):
        bentoml_deploy_version = get_bentoml_deploy_version()
        self._conda_env = CondaEnv(bentoml_version=bentoml_deploy_version)
        self._pip_dependencies = ["bentoml=={}".format(bentoml_deploy_version)]
        self._python_version = PYTHON_VERSION

        self._setup_sh = None

    def get_conda_env_name(self):
        return self._conda_env.get_name()

    def set_conda_env_name(self, name):
        self._conda_env.set_name(name)

    def add_conda_channels(self, channels):
        if not isinstance(channels, list):
            channels = [channels]
        self._conda_env.add_channels(channels)

    def add_conda_dependencies(self, conda_dependencies):
        if not isinstance(conda_dependencies, list):
            conda_dependencies = [conda_dependencies]
        self._conda_env.add_conda_dependencies(conda_dependencies)

    def add_conda_pip_dependencies(self, pip_dependencies):
        if not isinstance(pip_dependencies, list):
            pip_dependencies = [pip_dependencies]
        self._conda_env.add_pip_dependencies(pip_dependencies)

    def add_handler_dependencies(self, handler_dependencies):
        if not isinstance(handler_dependencies, list):
            handler_dependencies = [handler_dependencies]
        self._pip_dependencies += handler_dependencies

    def set_setup_sh(self, setup_sh_path_or_content):
        setup_sh_file = Path(setup_sh_path_or_content)

        if setup_sh_file.is_file():
            with setup_sh_file.open("rb") as f:
                self._setup_sh = f.read()
        else:
            self._setup_sh = setup_sh_path_or_content.encode("utf-8")

    def add_pip_dependencies(self, pip_dependencies):
        if not isinstance(pip_dependencies, list):
            pip_dependencies = [pip_dependencies]
        self._pip_dependencies += pip_dependencies

    def set_requirements_txt(self, requirements_txt_path):
        requirements_txt_file = Path(requirements_txt_path)

        with requirements_txt_file.open("rb") as f:
            content = f.read()
            module_list = content.decode("utf-8").split("\n")
            self._pip_dependencies += module_list

    def save(self, path):
        conda_yml_file = os.path.join(path, "environment.yml")
        self._conda_env.write_to_yaml_file(conda_yml_file)

        requirements_txt_file = os.path.join(path, "requirements.txt")

        with open(requirements_txt_file, "wb") as f:
            pip_content = "\n".join(self._pip_dependencies).encode("utf-8")
            f.write(pip_content)

        if self._setup_sh:
            setup_sh_file = os.path.join(path, "setup.sh")
            with open(setup_sh_file, "wb") as f:
                f.write(self._setup_sh)

    @classmethod
    def from_dict(cls, env_dict):
        env = cls()

        if "setup_sh" in env_dict:
            env.set_setup_sh(env_dict["setup_sh"])

        if "requirements_txt" in env_dict:
            env.set_requirements_txt(env_dict["requirements_txt"])

        if "pip_dependencies" in env_dict:
            env.add_pip_dependencies(env_dict["pip_dependencies"])

        if "conda_channels" in env_dict:
            env.add_conda_channels(env_dict["conda_channels"])

        if "conda_dependencies" in env_dict:
            env.add_conda_dependencies(env_dict["conda_dependencies"])

        if "conda_pip_dependencies" in env_dict:
            env.add_conda_pip_dependencies(env_dict["conda_pip_dependencies"])

        return env

    def to_dict(self):
        env_dict = dict()

        if self._setup_sh:
            env_dict["setup_sh"] = self._setup_sh

        if self._pip_dependencies:
            env_dict["pip_dependencies"] = self._pip_dependencies

        env_dict["conda_env"] = self._conda_env._conda_env

        env_dict["python_version"] = self._python_version
        return env_dict
