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

import re

from six.moves.urllib.parse import urlparse, uses_netloc, uses_params, uses_relative

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError, AttributeError):
    from pathlib2 import Path

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard('')


def is_url(url):
    try:
        return urlparse(url).scheme in _VALID_URLS
    except Exception:  # pylint:disable=broad-except
        return False


def isidentifier(s):
    """
    Return true if string is in a valid python identifier format:

    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    try:
        return s.isidentifier()
    except AttributeError:
        # str#isidentifier is only available in python 3
        return re.match(r"[A-Za-z_][A-Za-z_0-9]*\Z", s) is not None
