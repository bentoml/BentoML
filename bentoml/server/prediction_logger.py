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

# prediction log conf
PREDICTION_LOGGER_NAME = "PredictionLogger"
PREDICTION_LOG_FILE_PATH = "/tmp/logs"
PREDICTION_LOG_FILE = "/tmp/logs/bentoml_prediction.log"
PREDICTION_LOG_POS_FILE = "/tmp/logs/bentoml_prediction.log.pos"

# logging conf
LOG_FILE_MAX_SIZE = 100 * 1000 * 1000
LOG_FILE_NUM_BACKUPS = 10


def get_prediction_logger():
    """
    initialize logger for logging prediction results
    """

    # prediction.log json fields - request / result / time
    formatter = jsonlogger.JsonFormatter(
        '(service_name) (service_version) (api_name) (request_id) (request) (response) (asctime)')

    prediction_logger = logging.getLogger(PREDICTION_LOGGER_NAME)

    if not prediction_logger.handlers:
        prediction_logger.setLevel(logging.INFO)

        # Create log file and its dir, it not exist
        if os.path.isdir(PREDICTION_LOG_FILE_PATH) is False:
            os.mkdir(PREDICTION_LOG_FILE_PATH)
        if os.path.exists(PREDICTION_LOG_FILE) is False:
            open(PREDICTION_LOG_FILE, 'a').close()
        if os.path.exists(PREDICTION_LOG_POS_FILE) is False:
            open(PREDICTION_LOG_POS_FILE, "a").close()

        handler = RotatingFileHandler(filename=PREDICTION_LOG_FILE, maxBytes=LOG_FILE_MAX_SIZE,
                                      backupCount=LOG_FILE_NUM_BACKUPS)
        handler.setFormatter(formatter)
        prediction_logger.addHandler(handler)
        prediction_logger.propagate = False  # avoid duplicating the log in server logs

    return getLogger()


def getLogger():
    """
    Get prediction logger
    """
    return logging.getLogger(PREDICTION_LOGGER_NAME)


def parse_request(request):
    """
    Return request data for log prediction
    """
    # TODO: Handle images

    if request.content_type == 'application/json':
        return request.get_json()
    elif "image" in request.content_type:
        return {'data': 'dont handle'}
    elif "video" in request.content_type:
        return {'data': 'dont handle'}

    return {'data': request.get_data().decode('utf-8')}


def parse_response(response):
    """
    Return response prediction result for log prediction
    """
    return response.response


class PredictionLoggingMetaData():

    def __init__(self, service_name, service_version, api_name, request_id, asctime):
        self.service_name = service_name
        self.service_version = service_version
        self.api_name = api_name
        self.request_id = request_id
        self.asctime = asctime


def log_prediction(logger, metadata, request, response):
    """
    Log prediction result.
    """

    logger.info({
        "service_name": metadata['service_name'],
        "service_version": metadata['service_version'],
        "api_name": metadata['api_name'],
        "request_id": metadata['request_id'],
        "request": parse_request(request),
        "response": parse_response(response),
        "asctime": metadata['asctime'],
    })
