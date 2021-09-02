import logging
import os
from datetime import datetime
from pathlib import Path
from sys import version_info

import yaml
from simple_di import Provide, inject

from bentoml import __version__ as BENTOML_VERSION
from bentoml.exceptions import BentoMLConfigException

from ..configuration.containers import BentoMLContainer

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
                "name": artifact_name,
                "artifact_type": artifact.__class__.__name__,
                "metadata": artifact.metadata,
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
        self.config = yaml.safe_load(
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
            self.config["apis"] = _get_apis_list(bento_service)
            self.config["artifacts"] = _get_artifacts_list(bento_service)

    def write_to_path(self, path, filename="bentoml.yaml"):
        with open(Path(os.path.join(path, filename)), "r") as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, filepath):
        conf = cls()
        with open(filepath, "rb") as config_file:
            yml_content = config_file.read()
        conf.config = yaml.safe_load(yml_content)
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
                if not BENTOML_VERSION.startswith("0+untagged"):
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
