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

import os
import sys
import logging.config
from pathlib import Path

from bentoml import config
from bentoml.configuration import get_debug_mode


def get_logging_config_dict(logging_level, base_log_directory):
    conf = config("logging")  # proxy to logging section in bentoml config file

    LOG_FORMAT = conf.get("LOG_FORMAT")
    DEV_LOG_FORMAT = conf.get("DEV_LOG_FORMAT")

    PREDICTION_LOG_FILENAME = conf.get("prediction_log_filename")
    PREDICTION_LOG_JSON_FORMAT = conf.get("prediction_log_json_format")

    FEEDBACK_LOG_FILENAME = conf.get("feedback_log_filename")
    FEEDBACK_LOG_JSON_FORMAT = conf.get("feedback_log_json_format")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"format": LOG_FORMAT},
            "dev": {"format": DEV_LOG_FORMAT},
            "prediction": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "fmt": PREDICTION_LOG_JSON_FORMAT,
            },
            "feedback": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "fmt": FEEDBACK_LOG_JSON_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "level": logging_level,
                "formatter": "console",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
            "local": {
                "level": logging_level,
                "formatter": "dev",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": os.path.join(base_log_directory, "active.log"),
                "maxBytes": 100 * 1000 * 1000,
                "backupCount": 2,
            },
            "prediction": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "prediction",
                "level": "INFO",
                "filename": os.path.join(base_log_directory, PREDICTION_LOG_FILENAME),
                "maxBytes": 100 * 1000 * 1000,
                "backupCount": 10,
            },
            "feedback": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "feedback",
                "level": "INFO",
                "filename": os.path.join(base_log_directory, FEEDBACK_LOG_FILENAME),
                "maxBytes": 100 * 1000 * 1000,
                "backupCount": 10,
            },
        },
        "loggers": {
            "bentoml": {
                "handlers": ["console", "local"],
                "level": logging_level,
                "propagate": False,
            },
            "bentoml.prediction": {
                "handlers": ["prediction", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "bentoml.feedback": {
                "handlers": ["feedback", "console"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


def configure_logging(logging_level=None):
    if logging_level is None:
        logging_level = config("logging").get("LEVEL").upper()
        if "LOGGING_LEVEL" in config("logging"):
            # Support legacy config name e.g. BENTOML__LOGGING__LOGGING_LEVEL=debug
            logging_level = config("logging").get("LOGGING_LEVEL").upper()

    if get_debug_mode():
        logging_level = logging.getLevelName(logging.DEBUG)

    base_log_dir = os.path.expanduser(config("logging").get("BASE_LOG_DIR"))
    Path(base_log_dir).mkdir(parents=True, exist_ok=True)
    logging_config = get_logging_config_dict(logging_level, base_log_dir)
    logging.config.dictConfig(logging_config)
