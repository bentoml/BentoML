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
import logging
from datetime import datetime

from ruamel.yaml import YAML

from bentoml import __version__ as BENTOML_VERSION
from bentoml.configuration import get_bentoml_deploy_version
from bentoml.utils import Path

BENTOML_CONFIG_YAML_TEPMLATE = """\
version: {bentoml_version}
kind: {kind}
metadata:
    created_at: {created_at}
"""

LOG = logging.getLogger(__name__)


class BentoArchiveConfig(object):
    def __init__(self, kind="BentoService"):
        self.kind = kind
        self._yaml = YAML()
        self._yaml.default_flow_style = False
        self.config = self._yaml.load(
            BENTOML_CONFIG_YAML_TEPMLATE.format(
                kind=self.kind,
                bentoml_version=get_bentoml_deploy_version(),
                created_at=str(datetime.utcnow()),
            )
        )

    def write_to_path(self, path, filename="bentoml.yml"):
        return self._yaml.dump(self.config, Path(os.path.join(path, filename)))

    @classmethod
    def load(cls, filepath):
        conf = cls()
        with open(filepath, "rb") as config_file:
            yml_content = config_file.read()
        conf.config = conf._yaml.load(yml_content)

        if conf["version"] != BENTOML_VERSION:
            msg = (
                "BentoArchive version mismatch: loading archive bundled in version "
                "{},  but loading from version {}".format(
                    conf["version"], BENTOML_VERSION
                )
            )

            # If major version is different, then there could be incompatible API
            # changes. Raise error in this case.
            if conf["version"].split(".")[0] != BENTOML_VERSION.split(".")[0]:
                if not BENTOML_VERSION.startswith('0+untagged'):
                    raise ValueError(msg)
                else:
                    LOG.warning(msg)
            else:  # Otherwise just show a warning.
                LOG.warning(msg)

        return conf

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, item):
        return item in self.config
