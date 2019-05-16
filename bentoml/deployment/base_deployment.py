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

from bentoml.archive import load


class Deployment(object):
    """Deployment is spec for describing what actions deployment should have
    to interact with BentoML cli and BentoML service archive.
    """

    def __init__(self, archive_path):
        self.bento_service = load(archive_path)
        self.archive_path = archive_path

    def deploy(self):
        """Deploy bentoml service.

        :return: Boolean, True if success
        """
        raise NotImplementedError

    def check_status(self):
        """Check deployment status

        :params
        :return: Boolean, True if success Status Message String
        """
        raise NotImplementedError

    def delete(self):
        """Delete deployment, if deployment is active
        :return: Boolean, True if success
        """
        raise NotImplementedError
