import logging
import os

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException

logger = logging.getLogger(__name__)


COREMLMODEL_FILE_EXTENSION = ".mlmodel"


class CoreMLModel(BaseArtifact):
    """
    Artifact class for saving/loading coreml.models.MLModel objects
    with :obj:`coremltools.models.MLModel.save` and :obj:`coremltools.models.MLModel(path)`.

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: coremltools package required for CoreMLModel
        InvalidArgument: invalid argument type, model being packed must be instance of
            coremltools.models.MLModel

    Example usage::

        # train.py
        import coremltools as ct
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                ...

            def forward(self, x):
                ...

        net = Net()
        # Train model with data, then convert to CoreML.
        model = ct.convert(net, ...)

        # bento_service.py
        TODO:
        import bentoml
        import PIL.Image
        from bentoml.adapters import ImageInput
        from bentoml.frameworks.coreml import CoreMLModel

        @bentoml.env(infer_pip_packages=True)
        @bentoml.artifacts([CoreMLModel('model')])
        class CoreMLModelService(bentoml.BentoService):

            @bentoml.api(input=ImageInput(), batch=True)
            def predict(self, imgs):
                outputs = [self.artifacts.model(PIL.Image.fromarray(_.astype("uint8")))
                           for img in imgs]
                return outputs

        # bento_packer.py
        TODO:
        svc = CoreMLModelService()
        # Pytorch model can be packed directly.
        svc.pack('model', model)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._model = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + COREMLMODEL_FILE_EXTENSION)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
        try:
            import coremltools
        except ImportError:
            raise MissingDependencyException(
                "coremltools>=4.0b2 package is required to use CoreMLModel"
            )

        if not isinstance(model, coremltools.models.MLModel):
            raise InvalidArgument(
                "CoreMLModel can only pack type 'coremltools.models.MLModel'"
            )

        self._model = model
        return self

    @classmethod
    def load(cls, path):
        try:
            import coremltools
        except ImportError:
            raise MissingDependencyException(
                "coremltools package is required to use CoreMLModel"
            )

        model = coremltools.models.MLModel(self._file_path(path))

        if not isinstance(model, coremltools.models.MLModel):
            raise InvalidArgument(
                "Expecting CoreMLModel loaded object type to be "
                "'coremltools.models.MLModel' but actually it is {}".format(type(model))
            )

        return self.pack(model)

    def save(self, dst):
        self._model.save(self._file_path(dst))
