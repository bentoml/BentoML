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
