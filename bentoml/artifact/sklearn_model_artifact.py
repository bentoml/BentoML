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

import os

from bentoml.artifact import BentoServiceArtifact
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

    Example usage:

    >>> from sklearn import svm
    >>>
    >>> model_to_save = svm.SVC(gamma='scale')
    >>> # ... training model, etc.
    >>>
    >>> import bentoml
    >>> from bentoml.artifact import SklearnModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.env(auto_pip_dependencies=True)
    >>> @bentoml.artifacts([SklearnModelArtifact('model')])
    >>> class SklearnModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput())
    >>>     def predict(self, df):
    >>>         result = self.artifacts.model.predict(df)
    >>>         return result
    >>>
    >>> svc = SklearnModelService()
    >>>
    >>> # Pack directly with sklearn model object
    >>> svc.pack('model', model_to_save)
    """

    def __init__(self, name, pickle_extension=".pkl"):
        super(SklearnModelArtifact, self).__init__(name)

        self._pickle_extension = pickle_extension
        self._model = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._pickle_extension)

    def pack(self, sklearn_model):  # pylint:disable=arguments-differ
        self._model = sklearn_model
        return self

    def load(self, path):
        joblib = _import_joblib_module()

        model_file_path = self._model_file_path(path)
        sklearn_model = joblib.load(model_file_path, mmap_mode='r')
        return self.pack(sklearn_model)

    def get(self):
        return self._model

    def save(self, dst):
        joblib = _import_joblib_module()

        joblib.dump(self._model, self._model_file_path(dst))

    def set_dependencies(self, env):
        env.add_pip_dependencies_if_missing(['scikit-learn'])
