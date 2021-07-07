import importlib
import os

from bentoml.exceptions import MissingDependencyException
from bentoml.service.env import BentoServiceEnv
from bentoml.service.artifacts import BentoServiceArtifact

EVALML_MODEL_PICKLE_EXTENTION = ".pkl"


class EvalMLModelArtifact(BentoServiceArtifact):
    """
    Artifact class  for saving/loading EvalML models

    Args:
        name (str): Name for the artifact

    Raises:
        MissingDependencyException: evalml package is required for EvalMLModelArtifact

    Example usage:

    >>> # import the EvalML base pipeline class corresponding to your ML task
    >>> from evalml.pipelines import BinaryClassificationPipeline
    >>> # create an EvalML pipeline instance with desired components
    >>> model_to_save = BinaryClassificationPipeline(
    >>>     ['Imputer', 'One Hot Encoder', 'Random Forest Classifier'])
    >>> # Train model with data using model_to_save.fit(X, y)
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.sklearn import EvalMLModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([EvalMLModelArtifact('EvalML pipeline')])
    >>> class EvalMLModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         result = self.artifacts.model.predict(df)
    >>>         return result
    >>>
    >>> svc = EvalMLModelService()
    >>>
    >>> # Pack directly with EvalML model object
    >>> svc.pack('model', model_to_save)
    >>> svc.save()
    """

    def __init__(self, name):
        super().__init__(name)

        self._model = None

    def _validate_package(self):
        try:
            importlib.import_module('evalml')
        except ImportError:
            raise MissingDependencyException(
                "Package 'evalml' is required to use EvalMLModelArtifact"
            )

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + EVALML_MODEL_PICKLE_EXTENTION)

    def pack(self, evalml_model, metadata=None):  # pylint:disable=arguments-renamed
        self._validate_package()
        self._model = evalml_model
        return self

    def load(self, path):
        self._validate_package()
        model_file_path = self._model_file_path(path)
        evalml_pipelines_module = importlib.import_module('evalml.pipelines')
        model = evalml_pipelines_module.PipelineBase.load(model_file_path)
        self.pack(model)
        return model

    def get(self):
        return self._model

    def save(self, dst):
        self._validate_package()
        model_file_path = self._model_file_path(dst)
        self._model.save(model_file_path)

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(['evalml'])
