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

from bentoml import handlers
from bentoml.version import __version__

from bentoml.service import BentoService, api_decorator as api, \
    env_decorator as env, artifacts_decorator as artifacts, \
    ver_decorator as ver
from bentoml.server import metrics
from bentoml.archive import save, load

__all__ = [
    '__version__', 'api', 'env', 'artifacts', 'BentoService', 'save', 'load', 'handlers', 'metrics'
]
