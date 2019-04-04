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

from bentoml.artifacts import Artifact


class TfKerasModelArtifact(Artifact):
    """
    Abstraction for saving/loading tensorflow keras model
    """

    def __init__(self, name, model_extension='.h5'):
        self.model = None
        self._model_extension = model_extension
        super(TfKerasModelArtifact, self).__init__(name)

    def model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        self.model = model

    def get(self):
        return self.model

    def load(self, base_path):
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            raise ImportError("tensorflow package is required to use TfKerasModelArtifact")
        self.model = load_model(self.model_file_path(base_path))

    def save(self, base_path):
        if not self.model:
            raise RuntimeError("Must 'pack' artifact before 'save'.")

        self.model.save(self.model_file_path(base_path))
