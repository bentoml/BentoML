import os
from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class GluonModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading gluon models

    Args:
        name (str): Name for the artifact

    Raises:
        MissingDependencyError: mxnet package is required for GluonModelArtifact

    Example usage:

    >>> from bentoml import env, artifacts, api, BentoService
    >>> from bentoml.adapters import JsonInput
    >>> from bentoml.frameworks.gluon import GluonModelArtifact
    >>> import mxnet as mx
    >>>
    >>> @env(infer_pip_packages=True)
    >>> @artifacts([GluonModelArtifact('model')])
    >>> class GluonClassifier(BentoService):
    >>>     @api(input=JsonInput(), batch=False)
    >>>     def predict(self, request):
    >>>         nd_input = mx.nd.array(request['input'])
    >>>     return self.artifacts.model(nd_input).asnumpy()
    >>>
    >>> svc = GluonClassifier()
    >>> svc.pack('model', model_to_save)
    >>> svc.save()
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._model = None

    def pack(self, model, metadata: dict = None):  # pylint: disable=unused-argument
        try:
            import mxnet  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "mxnet package is required to use GluonModelArtifact"
            )
        self._model = model
        return self

    def load(self, path):
        try:
            from mxnet import gluon  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "mxnet package is required to use GluonModelArtifact"
            )

        prefix = self._model_file_path(path)
        model = gluon.nn.SymbolBlock.imports(
            "{}-symbol.json".format(prefix), ["data"], "{}-0000.params".format(prefix)
        )
        return self.pack(model)

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def save(self, dst):
        self._model.export(self._model_file_path(dst))

    def get(self):
        return self._model

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(["mxnet"])
