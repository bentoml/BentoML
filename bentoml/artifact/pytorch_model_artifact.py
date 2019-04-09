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

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class PytorchModelArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading objects with torch.save and torch.load
    """

    def __init__(self, name, pickle_module='dill', file_extension='.pt'):
        super(PytorchModelArtifact, self).__init__(name)
        self._file_extension = file_extension
        if isinstance(pickle_module, string_types):
            self._pickle = __import__(pickle_module)
        else:
            self._pickle = pickle_module

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _PytorchModelArtifactInstance(self, model)

    def load(self, path):
        try:
            import torch
        except ImportError:
            raise ImportError("torch package is required to use PytorchModelArtifact")

        model = torch.load(self._file_path(path), pickle_module=self._pickle)
        return self.pack(model)


class _PytorchModelArtifactInstance(ArtifactInstance):

    def __init__(self, spec, model):
        super(_PytorchModelArtifactInstance, self).__init__(spec)

        try:
            import torch
        except ImportError:
            raise ImportError("torch package is required to use PytorchModelArtifact")

        if not isinstance(model, torch.nn.Module):
            raise TypeError("PytorchModelArtifact can only pack type 'torch.nn.Module'")

        self._model = model

    def get(self):
        return self._model

    def save(self, dst):
        try:
            import torch
        except ImportError:
            raise ImportError("torch package is required to use PytorchModelArtifact")

        return torch.save(self._model, self.spec._file_path(dst), pickle_module=self.spec._pickle)
