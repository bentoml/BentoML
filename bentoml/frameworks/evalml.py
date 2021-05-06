import importlib
import os

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact


class EvalMLModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading EvalML models

    Args:
        name (str): Name for the artifact
        pickle_extension (str): The extension format for pickled file

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

    def __init__(self, name, pickle_extension=".pkl"):
        super(EvalMLModelArtifact, self).__init__(name)

        self._pickle_extension = pickle_extension
        self._model = None
        self._pipeline_base_class = None

    def _validate_package(self):
        try:
            evalml_module = importlib.import_module('evalml')  # noqa # pylint: disable=unused-variable
        except ImportError:
            raise MissingDependencyException(
                "Package 'evalml' is required to use EvalMLModelArtifact"
            )

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._pickle_extension)

    def pack(self, evalml_model, metadata=None):  # pylint:disable=arguments-differ
        self._validate_package()
        self._model = evalml_model
        return self

    def load(self, path):
        self._validate_package()
        model_file_path = self._model_file_path(path)
        if self._pipeline_base_class is None:
            evalml_pipelines_module_name = 'evalml.pipelines'
            evalml_pipelines_module = importlib.import_module(
                evalml_pipelines_module_name)
            self._pipeline_base_class = evalml_pipelines_module.PipelineBase
        model = self._pipeline_base_class.load(model_file_path)
        self.pack(model)
        return model

    def get(self):
        return self._model

    def save(self, dst):
        self._validate_package()
        model_file_path = self._model_file_path(dst)
        self._model.save(model_file_path)

    def set_dependencies(self, env):
        env.add_pip_packages(['evalml'])
