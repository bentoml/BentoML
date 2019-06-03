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
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

FEEDBACK_LOGGER_NAME = "FeedbackLogger"
FEEDBACK_LOG_FILE_PATH = "/tmp/logs"
FEEDBACK_LOG_FILE = "/tmp/logs/bentoml_feedback.log"


def get_feedback_logger():
    formatter = jsonlogger.JsonFormatter('(request_id) (result)')
    feedback_logger = logging.getLogger(FEEDBACK_LOGGER_NAME)

    if not feedback_logger.handlers:
        feedback_logger.setLevel(logging.INFO)

        # Create log file and its dir, it not exist
        if os.path.isdir(FEEDBACK_LOG_FILE_PATH) is False:
            os.mkdir(FEEDBACK_LOG_FILE_PATH)
        if os.path.exists(FEEDBACK_LOG_FILE) is False:
            open(FEEDBACK_LOG_FILE, 'a').close()

        handler = RotatingFileHandler(filename=FEEDBACK_LOG_FILE)
        handler.setFormatter(formatter)
        feedback_logger.addHandler(handler)

    return logging.getLogger(FEEDBACK_LOGGER_NAME)


def log_feedback(logger, data):
    logger.info(data)
