import os
from six import string_types

from bentoml.artifacts import Artifact


class PytorchModelArtifact(Artifact):
    """
    Abstraction for saving/loading objects with torch.save and torch.load
    """

    def __init__(self, name, pickle_module='dill', file_extension='.pt'):
        self.model = None
        self._file_extension = file_extension
        if isinstance(pickle_module, string_types):
            self._pickle = __import__(pickle_module)
        else:
            self._pickle = pickle_module
        super(PytorchModelArtifact, self).__init__(name)

    def file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        import torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError("PytorchModelArtifact can only pack type 'torch.nn.Module'")

        self.model = model

    def get(self):
        return self.model

    def save(self, base_path):
        if not self.model:
            raise RuntimeError("Must 'pack' artifact before 'save'.")

        import torch
        torch.save(self.model, self.file_path(base_path), pickle_module=self._pickle)

    def load(self, base_path):
        import torch
        self.model = torch.load(self.file_path(base_path), pickle_module=self._pickle)
        self.model.eval()
