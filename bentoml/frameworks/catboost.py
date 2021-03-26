import os

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class CatBoostModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading CatBoost models.

    Args:
        name (string): name of the artifact
        model_type (string): Type of the model - 'classifier', 'regressor', 'catboost'
        model_extension (string): Extension name for saved CatBoost model. Default - "cbm"
        model_export_parameters (dict): Additional format-dependent parameters. Default - None
        model_pool: The dataset previously used for training. See catboost.Pool for more details. Default - None.

    Raises:
        MissingDependencyException: catboost package is required for using
            CatBoostModelArtifact.
        InvalidArgument: invalid argument type, model being packed must be instance of
            catboost.core.CatBoost

    Example usage:
    
    >>> import numpy as np
    >>> from catboost import CatBoostClassifier
    >>>
    >>> # Prepare data
    >>> train_data = np.random.randint(0, 100, size=(100, 10))
    >>> train_labels = np.random.randint(0, 2, size=(100))
    >>>
    >>> # train model
    >>> clf = CatBoostClassifier(loss_function='Logloss'. ...)
    >>> clf.fit(train_data, train_labels)
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import DataframeInput
    >>> from bentoml.frameworks.catboost import CatBoostModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @@bentoml.artifacts([CatBoostModelArtifact("model", model_type="classifier")])
    >>> class CatBoostModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         return self.artifacts.model.predict(df)
    >>>
    >>> clf_service = CatBoostModelService()
    >>> clf_service.pack('model', clf)
    """

    def __init__(
        self,
        name,
        model_type="classifier",
        model_extension="cbm",
        model_export_parameters=None,
        model_pool=None,
    ):
        super(CatBoostModelArtifact, self).__init__(name)
        self._model_type = model_type
        self._model_extension = model_extension
        self._model_export_parameters = model_export_parameters
        self._model_pool = model_pool
        self._model = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + "." + self._model_extension)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
        try:
            import catboost
        except ImportError:
            raise MissingDependencyException(
                "catboost package is required for using CatBoostModelArtifact."
            )

        if not isinstance(model, catboost.core.CatBoost):
            raise InvalidArgument(
                "Expect `model` argument to be a `catboost.core.CatBoost` instance"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost
        except ImportError:
            raise MissingDependencyException(
                "catboost package is required for using CatBoostModelArtifact."
            )

        if self._model_type == "classifier":
            catboost = CatBoostClassifier()
        elif self._model_type == "regressor":
            catboost = CatBoostRegressor()
        else:
            catboost = CatBoost()

        catboost.load_model(self._model_file_path(path))
        return self.pack(catboost)

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(["catboost"])

    def save(self, dst):
        return self._model.save_model(
            self._model_file_path(dst),
            format=self._model_extension,
            export_parameters=self._model_export_parameters,
            pool=self._model_pool,
        )

    def get(self):
        return self._model
