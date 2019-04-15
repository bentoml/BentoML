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
import tempfile

from ruamel.yaml import YAML

from bentoml.version import __version__ as BENTOML_VERSION
from bentoml.utils.s3 import is_s3_url, download_from_s3
from bentoml.utils.exceptions import BentoMLException


def load_bentoml_config(path):
    try:
        with open(os.path.join(path, 'bentoml.yml'), 'r') as f:
            bentml_yml_content = f.read()
    except FileNotFoundError:
        raise ValueError("BentoML can't locate config file 'bentoml.yml'"
                         " in archive path: {}".format(path))

    yaml = YAML()
    # load bentoml.yml as config
    bentoml_config = yaml.load(bentml_yml_content)

    if bentoml_config['bentoml_version'] != BENTOML_VERSION:
        # TODO: warn about bentoml version mismatch
        pass

    # TODO: also check python version here

    return bentoml_config


def load_bento_service_class(archive_path):
    """
    Load a BentoService class from saved archive in given path

    :param archive_path: A BentoArchive path generated from BentoService.save call
        or the path to pip installed BentoArchive directory
    :return: BentoService class
    """
    if is_s3_url(archive_path):
        tempdir = tempfile.mkdtemp()
        download_from_s3(archive_path, tempdir)
        archive_path = tempdir

    config = load_bentoml_config(archive_path)

    # Load target module containing BentoService class from given path
    module_file_path = os.path.join(archive_path, config['service_name'], config['module_file'])
    if not os.path.isfile(module_file_path):
        # Try loading without service_name prefix, for loading from a installed PyPi
        module_file_path = os.path.join(archive_path, config['module_file'])

    if not os.path.isfile(module_file_path):
        raise BentoMLException('Can not locate module_file {} in archive {}'.format(
            config['module_file'], archive_path))

    # Prepend archive_path to sys.path for loading extra python dependencies
    sys.path.insert(0, archive_path)

    module_name = config['module_name']
    if module_name in sys.modules:
        # module already loaded, TODO: add warning
        module = sys.modules[module_name]
    elif sys.version_info >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif sys.version_info >= (3, 3):
        from importlib.machinery import SourceFileLoader
        # pylint:disable=deprecated-method
        module = SourceFileLoader(module_name, module_file_path).load_module(module_name)
        # pylint:enable=deprecated-method
    else:
        import imp
        module = imp.load_source(module_name, module_file_path)

    # Remove archive_path from sys.path to avoid import naming conflicts
    sys.path.remove(archive_path)

    model_service_class = module.__getattribute__(config['service_name'])
    # Set _bento_archive_path, which tells BentoService where to load its artifacts
    model_service_class._bento_archive_path = archive_path
    # Set cls._version, service instance can access it via svc.version
    model_service_class._bento_service_version = config['service_version']

    return model_service_class


def load(archive_path):
    svc_cls = load_bento_service_class(archive_path)
    return svc_cls.from_archive(archive_path)
