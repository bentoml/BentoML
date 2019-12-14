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

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.exceptions import MissingDependencyException


def _import_joblib_module():
    try:
        import joblib
    except ImportError:
        joblib = None

    if joblib is None:
        try:
            from sklearn.externals import joblib
        except ImportError:
            pass

    if joblib is None:
        raise MissingDependencyException(
            "sklearn module is required to use SklearnModelArtifact"
        )

    return joblib


class SklearnModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading scikit learn models using sklearn.externals.joblib

    Args:
        name (str): Name for the artifact
        pickle_extension (str): The extension format for pickled file

    Raises:
        MissingDependencyException: sklean package is required for SklearnModelArtifact
    """

    def __init__(self, name, pickle_extension=".pkl"):
        super(SklearnModelArtifact, self).__init__(name)

        self._pickle_extension = pickle_extension

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._pickle_extension)

    def pack(self, sklearn_model):  # pylint:disable=arguments-differ
        return _SklearnModelArtifactWrapper(self, sklearn_model)

    def load(self, path):
        joblib = _import_joblib_module()

        model_file_path = self._model_file_path(path)
        sklearn_model = joblib.load(model_file_path, mmap_mode='r')
        return self.pack(sklearn_model)


class _SklearnModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):
        super(_SklearnModelArtifactWrapper, self).__init__(spec)

        self._model = model

    def get(self):
        return self._model

    def save(self, dst):
        joblib = _import_joblib_module()

        joblib.dump(self._model, self.spec._model_file_path(dst))
