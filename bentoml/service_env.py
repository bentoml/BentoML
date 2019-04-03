# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from sys import version_info
from ruamel.yaml import YAML

from bentoml.utils import Path
from bentoml.version import __version__ as BENTOML_VERSION

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro)

CONDA_ENV_BASE_YAML = """
name: {name}
channels:
  - defaults
dependencies:
  - python={python_version}
  - pip:
    - bentoml=={bentoml_version}
"""

CONDA_ENV_DEFAULT_NAME = 'bentoml-custom-conda-env'


class CondaEnv(object):

    def __init__(self, name=CONDA_ENV_DEFAULT_NAME, python_version=PYTHON_VERSION,
                 bentoml_version=BENTOML_VERSION):
        self._yaml = YAML()
        self._yaml.default_flow_style = False
        self._conda_env = self._yaml.load(
            CONDA_ENV_BASE_YAML.format(name=name, python_version=python_version,
                                       bentoml_version=bentoml_version))

    def set_name(self, name):
        self._conda_env["name"] = name

    def get_name(self):
        return self._conda_env["name"]

    def add_conda_dependencies(self, extra_conda_dependencies):
        if not isinstance(extra_conda_dependencies, list):
            # TODO: add more validation here
            pass
        self._conda_env["dependencies"] += extra_conda_dependencies

    def add_pip_dependencies(self, extra_pip_dependencies):
        if not isinstance(extra_pip_dependencies, list):
            # TODO: add more validation here
            pass
        for dep in self._conda_env["dependencies"]:
            if isinstance(dep, dict) and 'pip' in dep:
                # there is already a pip list in conda_env, append extra deps
                dep += extra_pip_dependencies
                return self
        self._conda_env["dependencies"] += [{"pip": extra_pip_dependencies}]

    def add_channels(self, channels):
        if not isinstance(channels, list):
            # TODO: add more validation here
            pass
        self._conda_env["channels"] += channels

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

    def __init__(self):
        self._setup_sh = None
        self._conda_env = CondaEnv()
        self._requirements_txt = None

    def get_conda_env_name(self):
        return self._conda_env.get_name()

    def set_codna_env_name(self, name):
        self._conda_env.set_name(name)

    def add_conda_channels(self, channels):
        if not isinstance(channels, list):
            channels = [channels]

        self._conda_env.add_channels(channels)

    def add_conda_dependencies(self, conda_dependencies):
        if not isinstance(conda_dependencies, list):
            conda_dependencies = [conda_dependencies]
        self._conda_env.add_conda_dependencies(conda_dependencies)

    def add_pip_dependencies(self, pip_dependencies):
        if not isinstance(pip_dependencies, list):
            pip_dependencies = [pip_dependencies]
        self._conda_env.add_pip_dependencies(pip_dependencies)

    def set_setup_sh(self, setup_sh_str):
        if self._setup_sh:
            # TODO: show warning on overwriting setup.sh
            pass
        self._setup_sh = setup_sh_str

    def set_requirements_txt_file(self, requirements_txt_path):
        if self._requirements_txt:
            # TODO: show warning on overwriting requirements.txt
            pass
        requirements_txt_file = Path(requirements_txt_path)
        if requirements_txt_file.is_file():
            with requirements_txt_file.open() as f:
                self._requirements_txt = f.read()
        else:
            # TODO: show error, file not exist in given path
            pass

    def set_requirements_txt_str(self, requirements_txt_str):
        if self._requirements_txt:
            # TODO: show warning on overwriting requirements.txt
            pass
        self._requirements_txt = requirements_txt_str

    def save(self, path):
        conda_yml_file = os.path.join(path, 'environment.yml')
        self._conda_env.write_to_yaml_file(conda_yml_file)

        requirements_txt_file = os.path.join(path, 'requirements.txt')
        with open(requirements_txt_file, 'wb') as f:
            f.write(self._requirements_txt or b'')

        if self._setup_sh:
            setup_sh_file = os.path.join(path, 'setup.sh')
            with open(setup_sh_file, 'wb') as f:
                f.write(self._setup_sh)
