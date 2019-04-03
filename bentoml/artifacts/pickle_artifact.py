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
from six import string_types
import dill

from bentoml.artifacts import Artifact


class PickleArtifact(Artifact):
    """
    Abstraction for saving/loading python objects with customizable pickle_module
    """

    def __init__(self, name, pickle_module=dill, pickle_extension='.pkl'):
        self.obj = None
        self._pickle_extension = pickle_extension
        if isinstance(pickle_module, string_types):
            self._pickle = __import__(pickle_module)
        else:
            self._pickle = pickle_module
        super(PickleArtifact, self).__init__(name)

    def _pkl_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._pickle_extension)

    def pack(self, obj):  # pylint:disable=arguments-differ
        self.obj = obj

    def get(self):
        if not self.obj:
            # TODO: create special exception type for this?
            raise Exception("Must 'pack' artifact before 'get'.")

        return self.obj

    def load(self, base_path):
        with open(self._pkl_file_path(base_path), "rb") as pkl_file:
            self.obj = self._pickle.load(pkl_file)

    def save(self, base_path):
        if not self.obj:
            raise RuntimeError("Must 'pack' artifact before 'save'.")

        with open(self._pkl_file_path(base_path), "wb") as pkl_file:
            self._pickle.dump(self.obj, pkl_file)
        # TODO: catch error when file does not exist
