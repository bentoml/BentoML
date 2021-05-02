import os
import pickle

from bentoml.exceptions import ArtifactLoadingException, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.artifacts.common import PickleArtifact

class EvalMLModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading EvalML models

    Args:
        name (str): Name for the artifact
        pickle_extension (str): The extension format for pickled file
        pickle_module (str, module): pickle format to use: 'pickle' or 'cloudpickle'

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

    def __init__(self, name, pickle_extension=".pkl", pickle_module=pickle):
        super(EvalMLModelArtifact, self).__init__(name)

        self._pickle_extension = pickle_extension
        self._pickle_module = pickle_module
        self._model = None

    def _validate_package(self):
        try:
            from evalml import __version__ as evalml_version  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "EvalML package is required to use EvalMLModelArtifact"
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
        try:
            pickle_artifact = PickleArtifact(self.name,
                                             pickle_module=self._pickle_module,
                                             pickle_extension=self._pickle_extension)
            pickle_artifact.load(path)
            model = pickle_artifact.get()
        except Exception:
            raise ArtifactLoadingException(
                f'File {model_file_path} is not unpickleable with module ' +
                f'{str(self._pickle_module)}.')
        self.pack(model)
        return model

    def get(self):
        return self._model

    def save(self, dst):
        self._validate_package()
        model_file_path = self._model_file_path(dst)
        try:
            pickle_artifact = PickleArtifact(self.name,
                                             pickle_module=self._pickle_module,
                                             pickle_extension=self._pickle_extension)
            pickle_artifact.pack(self._model)
            pickle_artifact.save(dst)
        except Exception:
            raise ArtifactLoadingException(
                f'File {model_file_path} is not unpickleable with module ' +
                f'{str(self._pickle_module)}.')

    def set_dependencies(self, env):
        env.add_pip_packages(['evalml'])
