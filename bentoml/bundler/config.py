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
from pathlib import Path

from ruamel.yaml import YAML

from bentoml import __version__ as BENTOML_VERSION
from bentoml.configuration import get_bentoml_deploy_version
from bentoml.utils import dump_to_yaml_str
from bentoml.proto.repository_pb2 import BentoServiceMetadata
from bentoml.exceptions import BentoMLConfigException


BENTOML_CONFIG_YAML_TEPMLATE = """\
version: {bentoml_version}
kind: {kind}
metadata:
    created_at: {created_at}
"""

logger = logging.getLogger(__name__)


def _get_apis_list(bento_service):
    result = []
    for api in bento_service.get_service_apis():
        result.append(
            {
                "name": api.name,
                "handler_type": api.handler.__class__.__name__,
                "docs": api.doc,
            }
        )
    return result


def _get_artifacts_list(bento_service):
    result = []
    for artifact_name in bento_service.artifacts:
        artifact_spec = bento_service.artifacts[artifact_name].spec
        result.append(
            {'name': artifact_name, 'artifact_type': artifact_spec.__class__.__name__}
        )
    return result


class SavedBundleConfig(object):
    def __init__(self, bento_service=None, kind="BentoService"):
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

        if bento_service is not None:
            self.config["metadata"].update(
                {
                    "service_name": bento_service.name,
                    "service_version": bento_service.version,
                }
            )
            self.config["env"] = bento_service.env.to_dict()
            self.config['apis'] = _get_apis_list(bento_service)
            self.config['artifacts'] = _get_artifacts_list(bento_service)

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
                "Saved BentoService bundle version mismatch: loading BentoServie "
                "bundle create with BentoML version {},  but loading from BentoML "
                "version {}".format(conf["version"], BENTOML_VERSION)
            )

            # If major version is different, then there could be incompatible API
            # changes. Raise error in this case.
            if conf["version"].split(".")[0] != BENTOML_VERSION.split(".")[0]:
                if not BENTOML_VERSION.startswith('0+untagged'):
                    raise BentoMLConfigException(msg)
                else:
                    logger.warning(msg)
            else:  # Otherwise just show a warning.
                logger.warning(msg)

        return conf

    def get_bento_service_metadata_pb(self):
        bento_service_metadata = BentoServiceMetadata()
        bento_service_metadata.name = self.config["metadata"]["service_name"]
        bento_service_metadata.version = self.config["metadata"]["service_version"]
        bento_service_metadata.created_at.FromDatetime(
            self.config["metadata"]["created_at"]
        )

        if "env" in self.config:
            if "setup_sh" in self.config["env"]:
                bento_service_metadata.env.setup_sh = self.config["env"]["setup_sh"]

            if "conda_env" in self.config["env"]:
                bento_service_metadata.env.conda_env = dump_to_yaml_str(
                    self.config["env"]["conda_env"]
                )

            if "pip_dependencies" in self.config["env"]:
                bento_service_metadata.env.pip_dependencies = "\n".join(
                    self.config["env"]["pip_dependencies"]
                )
            if "python_version" in self.config["env"]:
                bento_service_metadata.env.python_version = self.config["env"][
                    "python_version"
                ]

        if "apis" in self.config:
            for api_config in self.config["apis"]:
                api_metadata = BentoServiceMetadata.BentoServiceApi()
                if "name" in api_config:
                    api_metadata.name = api_config["name"]
                if "handler_type" in api_config:
                    api_metadata.handler_type = api_config["handler_type"]
                if "docs" in api_config:
                    api_metadata.docs = api_config["docs"]
                bento_service_metadata.apis.extend([api_metadata])

        if "artifacts" in self.config:
            for artifact_config in self.config["artifacts"]:
                artifact_metadata = BentoServiceMetadata.BentoArtifact()
                if "name" in artifact_config:
                    artifact_metadata.name = artifact_config["name"]
                if "artifact_type" in artifact_config:
                    artifact_metadata.artifact_type = artifact_config["artifact_type"]
                bento_service_metadata.artifacts.extend([artifact_metadata])

        return bento_service_metadata

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, item):
        return item in self.config
