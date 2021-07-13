import os

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

LIGHTGBM_MODEL_EXTENTION = ".txt"


class LightGBMModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving and loading LightGBM Model

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: lightgbm package is required for using
            LightGBMModelArtifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            lightgbm.Booster

    Example usage:

    >>> import lightgbm as lgb
    >>> # Prepare data
    >>> train_data = lgb.Dataset(...)
    >>> # train model
    >>> model = lgb.train(train_set=train_data, ...)
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.lightgbm import LightGBMModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.artifacts([LightGBMModelArtifact('model')])
    >>> @bentoml.env(infer_pip_packages=True)
    >>> class LgbModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         return self.artifacts.model.predict(df)
    >>>
    >>> svc = LgbModelService()
    >>> svc.pack('model', model)
    """

    def __init__(self, name):
        super(LightGBMModelArtifact, self).__init__(name)
        self._model = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + LIGHTGBM_MODEL_EXTENTION)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
        try:
            import lightgbm as lgb
        except ImportError:
            raise MissingDependencyException(
                "lightgbm package is required to use LightGBMModelArtifact"
            )

        if not isinstance(model, lgb.Booster):
            raise InvalidArgument(
                "Expect `model` argument to be a `lightgbm.Booster` instance"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            import lightgbm as lgb
        except ImportError:
            raise MissingDependencyException(
                "lightgbm package is required to use LightGBMModelArtifact"
            )
        bst = lgb.Booster(model_file=self._model_file_path(path))

        return self.pack(bst)

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(['lightgbm'])

    def save(self, dst):
        return self._model.save_model(self._model_file_path(dst))

    def get(self):
        return self._model
