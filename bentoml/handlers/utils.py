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

import sys
import argparse


def merge_dicts(x, y):
    """
    Merge 2 dictionaries into one. The second directory override first dictionary's key
    """
    y = y if y is not None else {}
    temp = x.copy()
    temp.update(y)
    return temp


def generate_cli_default_parser():
    """
    Create default parser for CLI tool.  With input and output option
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('-o', '--output', default='json')

    return parser
