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
import sys

from ruamel.yaml import YAML

from bentoml.service import BentoService
from bentoml.version import __version__ as BENTOML_VERSION


class _LoadedBentoServiceWrapper(BentoService):

    def __init__(self, model_service_class, path, config):
        super(_LoadedBentoServiceWrapper, self).__init__()
        self.path = path
        self.config = config
        self.model_service = model_service_class()
        self.loaded = False
        self._wrap_api_funcs()

    def load(self, path=None):
        if path is not None:
            # TODO: warn user path is ignored when using pip installed bentoML model
            pass
        self.model_service.load(self.path)
        self.loaded = True

    @property
    def apis(self):
        return self.model_service._apis

    @property
    def name(self):
        return self.config['model_name']

    @property
    def version(self):
        return self.config['model_version']

    def _wrap_api_funcs(self):
        """
        Add target ModelService's API methods to the returned wrapper class
        """
        for api in self.apis:
            setattr(self, api.name,
                    self.model_service.__getattribute__(api.name).__get__(self.model_service))


def load_bentoml_config(path):
    try:
        with open(os.path.join(path, 'bentoml.yml'), 'r') as f:
            bentml_yml_content = f.read()
    except FileNotFoundError:
        raise ValueError("Bentoml can't locate model config file 'bentoml.yml' "
                         " in the give path: {}".format(path))

    yaml = YAML()
    # load bentoml.yml as config
    bentoml_config = yaml.load(bentml_yml_content)

    if bentoml_config['bentoml_version'] != BENTOML_VERSION:
        # TODO: warn about bentoml version mismatch
        pass

    # TODO: also check python version here

    return bentoml_config


def load(path, lazy_load=False):
    """
    Load a BentoService or BentoModel from saved archive in given path

    :param path: A BentoArchive path generated from BentoService.save call
    :return: BentoService
    """
    config = load_bentoml_config(path)

    # Load target module containing BentoService class from given path
    module_file_path = os.path.join(path, config['model_name'], config['module_file'])

    if sys.version_info >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(config['module_name'], module_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif sys.version_info >= (3, 3):
        from importlib.machinery import SourceFileLoader
        # pylint:disable=deprecated-method
        module = SourceFileLoader(config['module_name'],
                                  module_file_path).load_module(config['module_name'])
        # pylint:enable=deprecated-method
    else:
        import imp
        module = imp.load_source(config['module_name'], module_file_path)

    model_service_class = module.__getattribute__(config['model_name'])
    loaded_model = _LoadedBentoServiceWrapper(model_service_class, path, config)

    if not lazy_load:
        loaded_model.load()

    return loaded_model
