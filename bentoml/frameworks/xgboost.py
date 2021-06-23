import os

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


XGBOOST_MODEL_EXTENSION = ".model"


class XgboostModelArtifact(BentoServiceArtifact):
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

    def __init__(self, name):
        super(XgboostModelArtifact, self).__init__(name)
        self._model = None

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(['xgboost'])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + XGBOOST_MODEL_EXTENSION)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
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

    def save(self, dst):
        return self._model.save_model(self._model_file_path(dst))

    def get(self):
        return self._model
