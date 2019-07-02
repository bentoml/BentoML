# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import dill
from six import string_types

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class PickleArtifact(ArtifactSpec):
    """Abstraction for saving/loading python objects with pickle serialization

    Args:
        name (str): Name for the artifact
        pickle_module (module|str): The python module will be used for pickle
            and unpickle artifact
        pickle_extension (str): The extension format for pickled file.
    """

    def __init__(self, name, pickle_module=dill, pickle_extension=".pkl"):
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
