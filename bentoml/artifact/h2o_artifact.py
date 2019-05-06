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
import shutil

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class H2oModelArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading objects with h2o.save_model and xgb.load_model
    """

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _H2oModelArtifactInstance(self, model)

    def load(self, path):
        try:
            import h2o
        except ImportError:
            raise ImportError("h2o package is required to use H2oModelArtifact")

        h2o.init()
        model = h2o.load_model(self._model_file_path(path))
        return self.pack(model)


class _H2oModelArtifactInstance(ArtifactInstance):

    def __init__(self, spec, model):
        super(_H2oModelArtifactInstance, self).__init__(spec)
        self._model = model

    def save(self, dst):
        try:
            import h2o
        except ImportError:
            raise ImportError("h2o package is required to use H2oModelArtifact")

        h2o_saved_path = h2o.save_model(model=self._model, path=dst, force=True)
        shutil.move(h2o_saved_path, self.spec._model_file_path(dst))
        return

    def get(self):
        return self._model
