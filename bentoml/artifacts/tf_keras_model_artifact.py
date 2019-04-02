import os

from bentoml.artifacts import Artifact


class TfKerasModelArtifact(Artifact):
    """
    Abstraction for saving/loading tensorflow keras model
    """

    def __init__(self, name, model_extension='.h5'):
        self.model = None
        self._model_extension = model_extension
        super(TfKerasModelArtifact, self).__init__(name)

    def model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def pack(self, model):
        self.model = model

    def get(self):
        return self.model

    def load(self, base_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(self.model_file_path(base_path))

    def save(self, base_path):
        if not self.model:
            raise RuntimeError("Must 'pack' artifact before 'save'.")

        self.model.save(self.model_file_path(base_path))
