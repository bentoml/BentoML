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

import dill
from six import string_types

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class PickleArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading python objects with pickle serialization
    """

    def __init__(self, name, pickle_module=dill, pickle_extension='.pkl'):
        super(PickleArtifact, self).__init__(name)

        self._pickle_extension = pickle_extension

        if isinstance(pickle_module, string_types):
            self._pickle = __import__(pickle_module)
        else:
            self._pickle = pickle_module

    def _pkl_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._pickle_extension)

    def pack(self, obj):  # pylint:disable=arguments-differ
        return _PickleArtifactInstance(self, obj)

    def load(self, path):
        with open(self._pkl_file_path(path), "rb") as pkl_file:
            obj = self._pickle.load(pkl_file)
        return self.pack(obj)


class _PickleArtifactInstance(ArtifactInstance):

    def __init__(self, spec, obj):
        super(_PickleArtifactInstance, self).__init__(spec)

        self._obj = obj

    def get(self):
        return self._obj

    def save(self, dst):
        with open(self.spec._pkl_file_path(dst), "wb") as pkl_file:
            self.spec._pickle.dump(self._obj, pkl_file)
