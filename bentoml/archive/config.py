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
from datetime import datetime

from ruamel.yaml import YAML

from bentoml.version import __version__ as BENTOML_VERSION
from bentoml.utils import Path

BENTOML_CONFIG_YAML_TEPMLATE = """\
version: {bentoml_version}
kind: {kind}
metadata:
    created_at: {created_at}
"""


class BentoArchiveConfig(object):

    def __init__(self, kind='BentoService'):
        self.kind = kind
        self._yaml = YAML()
        self._yaml.default_flow_style = False
        self.config = self._yaml.load(
            BENTOML_CONFIG_YAML_TEPMLATE.format(kind=self.kind, bentoml_version=BENTOML_VERSION,
                                                created_at=str(datetime.now())))

    def write_to_path(self, path, filename='bentoml.yml'):
        return self._yaml.dump(self.config, Path(os.path.join(path, filename)))

    @classmethod
    def load(cls, filepath):
        conf = cls()
        with open(filepath, 'rb') as config_file:
            yml_content = config_file.read()
        conf.config = conf._yaml.load(yml_content)

        if conf['version'] != BENTOML_VERSION:
            raise ValueError("BentoArchive version mismatch: loading archive bundled in version {},"
                             "but loading from version {}".format(conf['version'], BENTOML_VERSION))

        return conf

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value
