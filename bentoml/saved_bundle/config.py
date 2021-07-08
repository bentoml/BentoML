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

from datetime import datetime
import logging
import os
from pathlib import Path
from sys import version_info

from google.protobuf.struct_pb2 import Struct
from simple_di import Provide, inject

from bentoml import __version__ as BENTOML_VERSION
from bentoml.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLConfigException
from bentoml.utils import dump_to_yaml_str
from bentoml.utils.ruamel_yaml import YAML

BENTOML_CONFIG_YAML_TEMPLATE = """\
version: {bentoml_version}
kind: {kind}
metadata:
    created_at: {created_at}
"""

logger = logging.getLogger(__name__)
DEFAULT_MAX_LATENCY = 20000
DEFAULT_MAX_BATCH_SIZE = 4000
PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"


def _get_apis_list(bento_service):
    result = []
    for api in bento_service.inference_apis:
        api_obj = {
            "name": api.name,
            "docs": api.doc,
            "input_type": api.input_adapter.__class__.__name__,
            "output_type": api.output_adapter.__class__.__name__,
            "mb_max_batch_size": api.mb_max_batch_size,
            "mb_max_latency": api.mb_max_latency,
            "batch": api.batch,
            "route": api.route,
        }
        if api.input_adapter.config:
            api_obj["input_config"] = api.input_adapter.config
        if api.output_adapter.config:
            api_obj["output_config"] = api.output_adapter.config
        result.append(api_obj)
    return result


def _get_artifacts_list(bento_service):
    result = []
    for artifact_name, artifact in bento_service.artifacts.items():
        result.append(
            {
                'name': artifact_name,
                'artifact_type': artifact.__class__.__name__,
                'metadata': artifact.metadata,
            }
        )
    return result


class SavedBundleConfig(object):
    @inject
    def __init__(
        self,
        bento_service=None,
        kind="BentoService",
        bentoml_deployment_version: str = Provide[
            BentoMLContainer.bento_bundle_deployment_version
        ],
    ):
        self.kind = kind
        self._yaml = YAML()
        self._yaml.default_flow_style = False
        self.config = self._yaml.load(
            BENTOML_CONFIG_YAML_TEMPLATE.format(
                kind=self.kind,
                bentoml_version=bentoml_deployment_version,
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
        ver = str(conf["version"])
        py_ver = conf.config["env"]["python_version"]

        if ver != BENTOML_VERSION:
            msg = (
                "Saved BentoService bundle version mismatch: loading BentoService "
                "bundle create with BentoML version {}, but loading from BentoML "
                "version {}".format(conf["version"], BENTOML_VERSION)
            )

            # If major version is different, then there could be incompatible API
            # changes. Raise error in this case.
            if ver.split(".")[0] != BENTOML_VERSION.split(".")[0]:
                if not BENTOML_VERSION.startswith('0+untagged'):
                    raise BentoMLConfigException(msg)
                else:
                    logger.warning(msg)
            else:  # Otherwise just show a warning.
                logger.warning(msg)

        if py_ver != PYTHON_VERSION:
            logger.warning(
                f"Saved BentoService Python version mismatch: loading "
                f"BentoService bundle created with Python version {py_ver}, "
                f"but current environment version is {PYTHON_VERSION}."
            )

        return conf

    def get_bento_service_metadata_pb(self):
        from bentoml.yatai.proto.repository_pb2 import BentoServiceMetadata

        bento_service_metadata = BentoServiceMetadata()
        bento_service_metadata.name = self.config["metadata"]["service_name"]
        bento_service_metadata.version = self.config["metadata"]["service_version"]
        bento_service_metadata.created_at.FromDatetime(
            self.config["metadata"]["created_at"]
        )

        if "env" in self.config:
            env = self.config["env"]
            if "setup_sh" in env:
                bento_service_metadata.env.setup_sh = env["setup_sh"]

            if "conda_env" in env:
                bento_service_metadata.env.conda_env = dump_to_yaml_str(
                    env["conda_env"]
                )

            if "pip_packages" in env:
                for pip_package in env["pip_packages"]:
                    bento_service_metadata.env.pip_packages.append(pip_package)
            if "python_version" in env:
                bento_service_metadata.env.python_version = env["python_version"]
            if "docker_base_image" in env:
                bento_service_metadata.env.docker_base_image = env["docker_base_image"]
            if "requirements_txt" in env:
                bento_service_metadata.env.requirements_txt = env["requirements_txt"]

        if "apis" in self.config:
            for api_config in self.config["apis"]:
                if 'handler_type' in api_config:
                    # Convert handler type to input type for saved bundle created
                    # before version 0.8.0
                    input_type = api_config.get('handler_type')
                elif 'input_type' in api_config:
                    input_type = api_config.get('input_type')
                else:
                    input_type = "unknown"

                if 'output_type' in api_config:
                    output_type = api_config.get('output_type')
                else:
                    output_type = "DefaultOutput"

                api_metadata = BentoServiceMetadata.BentoServiceApi(
                    name=api_config["name"],
                    docs=api_config["docs"],
                    input_type=input_type,
                    output_type=output_type,
                )
                if "handler_config" in api_config:
                    # Supports viewing API input config info for saved bundle created
                    # before version 0.8.0
                    for k, v in api_config["handler_config"].items():
                        if k in {'mb_max_latency', 'mb_max_batch_size'}:
                            setattr(api_metadata, k, v)
                        else:
                            api_metadata.input_config[k] = v
                else:
                    if 'mb_max_latency' in api_config:
                        api_metadata.mb_max_latency = api_config["mb_max_latency"]
                    else:
                        api_metadata.mb_max_latency = DEFAULT_MAX_LATENCY

                    if 'mb_max_batch_size' in api_config:
                        api_metadata.mb_max_batch_size = api_config["mb_max_batch_size"]
                    else:
                        api_metadata.mb_max_batch_size = DEFAULT_MAX_BATCH_SIZE

                    if 'route' in api_config:
                        api_metadata.route = api_config["route"]
                    else:
                        # Use API name as the URL route when route config is missing,
                        # this is for backward compatibility for
                        # BentoML version <= 0.10.1
                        api_metadata.route = api_config["name"]

                if "input_config" in api_config:
                    for k, v in api_config["input_config"].items():
                        api_metadata.input_config[k] = v

                if "output_config" in api_config:
                    for k, v in api_config["output_config"].items():
                        api_metadata.output_config[k] = v
                api_metadata.batch = api_config.get("batch", False)
                bento_service_metadata.apis.extend([api_metadata])

        if "artifacts" in self.config:
            for artifact_config in self.config["artifacts"]:
                artifact_metadata = BentoServiceMetadata.BentoArtifact()
                if "name" in artifact_config:
                    artifact_metadata.name = artifact_config["name"]
                if "artifact_type" in artifact_config:
                    artifact_metadata.artifact_type = artifact_config["artifact_type"]
                if "metadata" in artifact_config:
                    if isinstance(artifact_config["metadata"], dict):
                        s = Struct()
                        s.update(artifact_config["metadata"])
                        artifact_metadata.metadata.CopyFrom(s)
                    else:
                        logger.warning(
                            "Tried to get non-dictionary metadata for artifact "
                            f"{artifact_metadata.name}. Ignoring metadata..."
                        )
                bento_service_metadata.artifacts.extend([artifact_metadata])

        return bento_service_metadata

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, item):
        return item in self.config
