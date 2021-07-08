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

import logging.config
import os
from pathlib import Path
import sys

from simple_di import Provide, inject

from bentoml.configuration import get_debug_mode
from bentoml.configuration.containers import BentoMLContainer


def get_logging_config_dict(
    logging_level: str,
    base_log_directory: str,
    console_logging_enabled: bool,
    file_logging_enabled: bool,
):
    MEGABYTES = 1024 * 1024

    handlers = {}
    bentoml_logger_handlers = []
    prediction_logger_handlers = []
    feedback_logger_handlers = []
    if console_logging_enabled:
        handlers.update(
            {
                "console": {
                    "level": logging_level,
                    "formatter": "console",
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                }
            }
        )
        bentoml_logger_handlers.append("console")
        prediction_logger_handlers.append("console")
        feedback_logger_handlers.append("console")
    if file_logging_enabled:
        handlers.update(
            {
                "local": {
                    "level": logging_level,
                    "formatter": "dev",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": os.path.join(base_log_directory, "active.log"),
                    "maxBytes": 100 * MEGABYTES,
                    "backupCount": 2,
                },
                "prediction": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "prediction",
                    "level": "INFO",
                    "filename": os.path.join(base_log_directory, "prediction.log"),
                    "maxBytes": 100 * MEGABYTES,
                    "backupCount": 10,
                },
                "feedback": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "feedback",
                    "level": "INFO",
                    "filename": os.path.join(base_log_directory, "feedback.log"),
                    "maxBytes": 100 * MEGABYTES,
                    "backupCount": 10,
                },
            }
        )
        bentoml_logger_handlers.append("local")
        prediction_logger_handlers.append("prediction")
        feedback_logger_handlers.append("feedback")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"format": "[%(asctime)s] %(levelname)s - %(message)s"},
            "dev": {
                "format": "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s"
                " - %(message)s"
            },
            "prediction": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"},
            "feedback": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"},
        },
        "handlers": handlers,
        "loggers": {
            "bentoml": {
                "handlers": bentoml_logger_handlers,
                "level": logging_level,
                "propagate": False,
            },
            "bentoml.prediction": {
                "handlers": prediction_logger_handlers,
                "level": "INFO",
                "propagate": False,
            },
            "bentoml.feedback": {
                "handlers": feedback_logger_handlers,
                "level": "INFO",
                "propagate": False,
            },
        },
    }


@inject
def configure_logging(
    logging_level: str = Provide[BentoMLContainer.config.logging.level],
    base_log_dir: str = Provide[BentoMLContainer.logging_file_directory],
    console_logging_enabled: bool = Provide[
        BentoMLContainer.config.logging.console.enabled
    ],
    file_logging_enabled: bool = Provide[BentoMLContainer.config.logging.file.enabled],
    advanced_enabled: bool = Provide[BentoMLContainer.config.logging.advanced.enabled],
    advanced_config: dict = Provide[BentoMLContainer.config.logging.advanced.config],
):
    Path(base_log_dir).mkdir(parents=True, exist_ok=True)
    if advanced_enabled:
        logging.config.dictConfig(advanced_config)
        logging.getLogger(__name__).debug(
            "Configured logging with advanced configuration, config=%s", advanced_config
        )
    else:
        if get_debug_mode():
            logging_level = logging.getLevelName(logging.DEBUG)

        logging_config = get_logging_config_dict(
            logging_level, base_log_dir, console_logging_enabled, file_logging_enabled
        )
        logging.config.dictConfig(logging_config)
        logging.getLogger(__name__).debug(
            "Configured logging with simple configuration, "
            "level=%s, directory=%s, console_enabled=%s, file_enabled=%s",
            logging_level,
            base_log_dir,
            console_logging_enabled,
            file_logging_enabled,
        )
