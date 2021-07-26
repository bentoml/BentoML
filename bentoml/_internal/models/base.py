import typing as t
from pathlib import Path

from ..types import MetadataType, PathType
from ..utils.ruamel_yaml import YAML

MT = t.TypeVar("MT")

H5_EXTENSION: str = ".h5"
HDF5_EXTENSION: str = ".hdf5"
JSON_EXTENSION: str = ".json"
PICKLE_EXTENSION: str = ".pkl"
PTH_EXTENSION: str = ".pth"
PT_EXTENSION: str = ".pt"
TXT_EXTENSION: str = ".txt"
YAML_EXTENSION: str = ".yaml"
YML_EXTENSION: str = ".yml"
MODEL_NAMESPACE: str = "bentoml_model"


class Model(object):
    """
    :class:`Model` is the base abstraction
     for describing the trained model serialization
     and deserialization process.

    Args:
        model (`MT`):
            Given model definition. Omit various type depending on given frameworks.
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata
    
    .. note::
        Make sure to add ``# noqa # pylint: disable=arguments-differ`` to :meth:`load` when implementing 
        newly integration or custom artifacts if the behaviour of ``load`` subclass takes different parameters
        
        .. code-block:: python

            from bentoml._internal.artifacts import Model
            
            class CustomModel(Model):
                def __init__(self, model, metadata=None):...

                @classmethod
                def load(cls, path: str, args1, args2):...  # noqa # pylint: disable=arguments-differ
            
    Example usage for creating a custom ``Model``::

        TODO:
    """

    def __init__(self: "Model", model: MT, metadata: t.Optional[MetadataType] = None):
        self._model = model
        self._metadata = metadata

    @property
    def metadata(self: "Model") -> t.Optional[MetadataType]:
        return self._metadata

    @classmethod
    def load(cls, path: PathType) -> MT:
        """
        Load saved model into memory.

        Args:
            path (`Union[str, os.PathLike]`):
                Given path to save artifacts metadata and objects.

        This will be used as a class method, interchangeable with
        :meth:`save` to load model during development pipeline.
        """
        raise NotImplementedError()

    def save(self: "Model", path: PathType) -> None:
        """
        Perform save instance to given path.

        Args:
            path (`Union[str, os.PathLike]`, or :obj:`~bentoml._internal.types.PathType`):
                Given path to save artifacts metadata and objects.

        Usually this can be used with :meth:`~bentoml._internal.artifacts.Model.load` to load
        model objects for development::

            # train.py
            model = MyPyTorchModel().train()  # type: torch.nn.Module
            ...
            from bentoml.pytorch import PyTorchModel
            PyTorchModel(model).save(".")
            pytorch_model = PyTorchModel.load(".")  # type: torch.nn.Module

        .. admonition:: current implementation

            Current implementation initialize base :meth:`save()` and :meth:`load()` in
            :code:`__getattribute__()` via wrapper. Since Python doesn't have support
            for method overloading, this ensures that model metadata will always be saved
            to given directory.
        """  # noqa # pylint: enable=line-too-long
        raise NotImplementedError()

    def __getattribute__(self: "Model", item: str) -> t.Any:
        if item == "save":

            def wrapped_save(*args, **kw):  # type: ignore
                path: PathType = args[0]  # save(self, path)
                if self.metadata:
                    yaml = YAML()
                    yaml.dump(
                        self.metadata, Path(path, f"{MODEL_NAMESPACE}{YML_EXTENSION}"),
                    )

                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_save
        elif item == "load":

            def wrapped_load(*args, **kw):  # type: ignore
                assert_msg: str = "`load()` requires positional `path`"
                assert "path" in args, assert_msg
                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_load
        else:
            return object.__getattribute__(self, item)
