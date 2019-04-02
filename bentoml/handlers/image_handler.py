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

from bentoml.handlers.base_handlers import RequestHandler, CliHandler


class ImageHandler(RequestHandler, CliHandler):
    """
    Image handler take input image and process them and return response or stdout.
    """

    @staticmethod
    def handle_request(request, func):
        raise NotImplementedError

    @staticmethod
    def handle_cli(options, func):
        raise NotImplementedError
