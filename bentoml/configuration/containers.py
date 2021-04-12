# Copyright 2020 Atalaya Tech, Inc.

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
import multiprocessing
import os

from deepmerge import always_merger
from dependency_injector import containers, providers
from schema import And, Or, Schema, SchemaError, Optional, Use

from bentoml.configuration import config
from bentoml.exceptions import BentoMLConfigException
from bentoml.utils.ruamel_yaml import YAML


LOGGER = logging.getLogger(__name__)

SCHEMA = Schema(
    {
        "bento_server": {
            "port": And(int, lambda port: port > 0),
            "workers": Or(And(int, lambda workers: workers > 0), None),
            "timeout": And(int, lambda timeout: timeout > 0),
            "max_request_size": And(int, lambda size: size > 0),
            "microbatch": {
                "enabled": bool,
                "workers": Or(And(int, lambda workers: workers > 0), None),
                "max_batch_size": Or(And(int, lambda size: size > 0), None),
                "max_latency": Or(And(int, lambda latency: latency > 0), None),
            },
            "ngrok": {"enabled": bool},
            "swagger": {"enabled": bool},
            "metrics": {"enabled": bool, "namespace": str},
            "feedback": {"enabled": bool},
            "logging": {"level": str},
        },
        "tracing": {
            "type": Or(
                And(str, Use(str.lower), lambda s: s in ('zipkin', 'jaeger')), None
            ),
            Optional("zipkin"): {"url": Or(str, None)},
            Optional("jaeger"): {"address": Or(str, None), "port": Or(int, None)},
        },
        "adapters": {"image_input": {"extensions": [str]}},
    }
)


class BentoMLConfiguration:
    def __init__(
        self,
        default_config_file: str = None,
        override_config_file: str = None,
        validate_schema: bool = True,
        legacy_compatibility: bool = True,
    ):
        # Default configuraiton
        if default_config_file is None:
            default_config_file = os.path.join(
                os.path.dirname(__file__), "default_bentoml.yml"
            )

        with open(default_config_file, "rb") as f:
            self.config = YAML().load(f.read())

        if validate_schema:
            try:
                SCHEMA.validate(self.config)
            except SchemaError as e:
                raise BentoMLConfigException(
                    "Default configuration 'default_bentoml.yml' does not"
                    " conform to the required schema."
                ) from e

        # Legacy configuration compatibility
        if legacy_compatibility:
            try:
                self.config["bento_server"]["port"] = config("apiserver").getint(
                    "default_port"
                )
                self.config["bento_server"]["workers"] = config("apiserver").getint(
                    "default_gunicorn_workers_count"
                )
                self.config["bento_server"]["max_request_size"] = config(
                    "apiserver"
                ).getint("default_max_request_size")

                if "default_max_batch_size" in config("marshal_server"):
                    self.config["bento_server"]["microbatch"][
                        "max_batch_size"
                    ] = config("marshal_server").getint("default_max_batch_size")

                if "default_max_latency" in config("marshal_server"):
                    self.config["bento_server"]["microbatch"]["max_latency"] = config(
                        "marshal_server"
                    ).getint("default_max_latency")

                self.config["bento_server"]["metrics"]["namespace"] = config(
                    "instrument"
                ).get("default_namespace")

                self.config["adapters"]["image_input"]["extensions"] = [
                    extension.strip()
                    for extension in config("apiserver")
                    .get("default_image_input_accept_file_extensions")
                    .split(",")
                ]
            except KeyError as e:
                raise BentoMLConfigException(
                    "Overriding a non-existent configuration key in compatibility mode."
                ) from e

            if validate_schema:
                try:
                    SCHEMA.validate(self.config)
                except SchemaError as e:
                    raise BentoMLConfigException(
                        "Configuration after applying legacy compatibility"
                        " does not conform to the required schema."
                    ) from e

        # User override configuration
        if override_config_file is not None:
            LOGGER.info("Applying user config override from %s" % override_config_file)
            if not os.path.exists(override_config_file):
                raise BentoMLConfigException(
                    f"Config file {override_config_file} not found"
                )

            with open(override_config_file, "rb") as f:
                override_config = YAML().load(f.read())
            always_merger.merge(self.config, override_config)

            if validate_schema:
                try:
                    SCHEMA.validate(self.config)
                except SchemaError as e:
                    raise BentoMLConfigException(
                        "Configuration after user override does not conform to"
                        " the required schema."
                    ) from e

    def override(self, keys: list, value):
        if keys is None:
            raise BentoMLConfigException("Configuration override key is None.")
        if len(keys) == 0:
            raise BentoMLConfigException("Configuration override key is empty.")
        if value is None:
            return

        c = self.config
        for key in keys[:-1]:
            if key not in c:
                raise BentoMLConfigException(
                    "Configuration override key is invalid, %s" % keys
                )
            c = c[key]
        c[keys[-1]] = value

        try:
            SCHEMA.validate(self.config)
        except SchemaError as e:
            raise BentoMLConfigException(
                "Configuration after applying override does not conform"
                " to the required schema, key=%s, value=%s." % (keys, value)
            ) from e

    def as_dict(self) -> dict:
        return self.config


class BentoMLContainer(containers.DeclarativeContainer):

    config = providers.Configuration(strict=True)

    api_server_workers = providers.Callable(
        lambda workers: workers if workers else (multiprocessing.cpu_count() // 2) + 1,
        config.bento_server.workers,
    )
