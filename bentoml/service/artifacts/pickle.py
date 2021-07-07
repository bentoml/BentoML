import os

from bentoml.service import BentoServiceArtifact
from bentoml.utils import cloudpickle


PICKLE_FILE_EXTENSION = ".pkl"


class PickleArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading python objects with pickle serialization

    Args:
        name (str): Name for the artifact
        pickle_module (module|str): The python module will be used for pickle
            and unpickle artifact, default pickle module in BentoML's fork of
            `cloudpickle`, which is identical to the Apache Spark fork
        pickle_extension (str): The extension format for pickled file.
    """

    def __init__(self, name, pickle_module=cloudpickle):
        super(PickleArtifact, self).__init__(name)

        self._obj = None

        if isinstance(pickle_module, str):
            self._pickle = __import__(pickle_module)
        else:
            self._pickle = pickle_module

    def _pkl_file_path(self, base_path):
        return os.path.join(base_path, self.name + PICKLE_FILE_EXTENSION)

    def pack(self, obj, metadata=None):  # pylint:disable=arguments-renamed
        self._obj = obj
        return self

    def load(self, path):
        with open(self._pkl_file_path(path), "rb") as pkl_file:
            obj = self._pickle.load(pkl_file)
        self.pack(obj)
        return self

    def get(self):
        return self._obj

    def save(self, dst):
        with open(self._pkl_file_path(dst), "wb") as pkl_file:
            self._pickle.dump(self._obj, pkl_file)
