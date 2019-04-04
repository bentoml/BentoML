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

# from flask import Request, Response, make_response, request
# import numpy as np
# import json
from bentoml.handlers.base_handlers import CliHandler, RequestHandler


class PytorchTensorHanlder(RequestHandler, CliHandler):
    """
    Tensor handlers for Pytorch models
    """

    @staticmethod
    def handle_request(request, func):
        raise NotImplementedError

    @staticmethod
    def handle_cli(options, func):
        raise NotImplementedError
