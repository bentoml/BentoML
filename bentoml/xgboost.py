# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import os

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException


class XgboostModelArtifact(ModelArtifact):
    """
    Artifact class for saving and loading Xgboost model

    Args:
        name (string): name of the artifact

    Raises:
        ImportError: xgboost package is required for using XgboostModelArtifact
        TypeError: invalid argument type, model being packed must be instance of
            xgboost.core.Booster

    Example usage:

    >>> import xgboost
    >>>
    >>> # prepare data
    >>> params = {... params}
    >>> dtrain = xgboost.DMatrix(...)
    >>>
    >>> # train model
    >>> model_to_save = xgboost.train(params=params, dtrain=dtrain)
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.xgboost import XgboostModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts(XgboostModelArtifact('model'))
    >>> class XGBoostModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         result = self.artifacts.model.predict(df)
    >>>         return result
    >>>
    >>> svc = XGBoostModelService()
    >>> # Pack xgboost model
    >>> svc.pack('model', model_to_save)
    """

    XGBOOST_EXTENSION = ".model"

    def __init__(self, name):
        super(XgboostModelArtifact, self).__init__(name)
        self._model = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self.XGBOOST_EXTENSION)

    def pack(self, model):  # pylint:disable=arguments-differ
        try:
            import xgboost as xgb
        except ImportError:
            raise MissingDependencyException(
                "xgboost package is required to use XgboostModelArtifact"
            )

        if not isinstance(model, xgb.core.Booster):
            raise InvalidArgument(
                "Expect `model` argument to be a `xgboost.core.Booster` instance"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            import xgboost as xgb
        except ImportError:
            raise MissingDependencyException(
                "xgboost package is required to use XgboostModelArtifact"
            )
        bst = xgb.Booster()
        bst.load_model(self._model_file_path(path))

        return self.pack(bst)

    def save(self, path):
        return self._model.save_model(self._model_file_path(path))

    def get(self):
        return self._model
