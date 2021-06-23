import os
import shutil

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class H2oModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving and loading h2o models using h2o.save_model and
    h2o.load_model

    Args:
        name (str): Name for this h2o artifact..

    Raises:
        MissingDependencyException: h2o package is required to use H2o model artifact

    Example usage:

    >>> import h2o
    >>> h2o.init()
    >>>
    >>> from h2o.estimators.deeplearning import H2ODeepLearningEstimator
    >>> model_to_save = H2ODeepLearningEstimator(...)
    >>> # train model with data
    >>> data = h2o.import_file(...)
    >>> model_to_save.train(...)
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.h2o import H2oModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.artifacts([H2oModelArtifact('model')])
    >>> @bentoml.env(infer_pip_packages=True)
    >>> class H2oModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         hf = h2o.H2OFrame(df)
    >>>         predictions = self.artifacts.model.predict(hf)
    >>>         return predictions.as_data_frame()
    >>>
    >>> svc = H2oModelService()
    >>>
    >>> svc.pack('model', model_to_save)
    """

    def __init__(self, name):
        super().__init__(name)

        self._model = None

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(['h2o'])
        env.add_conda_dependencies(['openjdk'])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
        self._model = model
        return self

    def load(self, path):
        try:
            import h2o
        except ImportError:
            raise MissingDependencyException(
                "h2o package is required to use H2oModelArtifact"
            )

        h2o.init()
        model = h2o.load_model(self._model_file_path(path))
        self._model = model
        return self

    def save(self, dst):
        try:
            import h2o
        except ImportError:
            raise MissingDependencyException(
                "h2o package is required to use H2oModelArtifact"
            )

        h2o_saved_path = h2o.save_model(model=self._model, path=dst, force=True)
        shutil.move(h2o_saved_path, self._model_file_path(dst))
        return

    def get(self):
        return self._model
