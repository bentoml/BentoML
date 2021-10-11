import functools
import logging
import os
import typing as t

import numpy as np
from simple_di import Provide, WrappedCallable
from simple_di import inject as _inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException, MissingDependencyException

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    try:
        import spacy
    except ImportError:
        raise MissingDependencyException(
            """spacy is required in order to use module `bentoml.spacy`, install
            spacy with `pip install spacy`. For more information, refer to
            https://spacy.io/usage
            """
        )
else:
    spacy = LazyLoader("spacy", globals(), "spacy")

logger = logging.getLogger(__name__)

inject: t.Callable[[WrappedCallable], WrappedCallable] = functools.partial(
    _inject, squeeze_none=False
)


def _get_model_info(
    tag: str,
    model_store: "ModelStore",
) -> t.Tuple["ModelInfo", str]:
    model_info = model_store.get(tag)
    if model_info.module != __name__:
        raise BentoMLException(
            f"Model {tag} was saved with module {model_info.module}, failed loading "
            f"with {__name__}."
        )
    model_file = os.path.join(model_info.path, f"{SAVE_NAMESPACE}.json")
    return model_info, model_file


@inject
def load(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "spacy.language.Language":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag('str'):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `spacy.language.Language` from BentoML modelstore

    Examples:
        import bentoml.spacy
        nlp = bentoml.spacy.load("my_spacy_model:latest")
    """  # noqa
    _, model_file = _get_model_info(tag, model_store)
    return spacy.load(model_file)


@inject
def save(
    name: str,
    model: "spacy.language.Language",
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name ('str'):
            Name for given model instance. This should pass Python identifier check.
        model ('spacy.language.Language"):
            Instance of model to be saved.
        metadata ('t.Union[None, t.Dict[str, t.Any]]'):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container

    Returns:
        tag ('str' with a format 'name:version') where 'name' is the defined name user sete
        for their models and version will be generated by BentoML.

    Examples:
        import spacy
        import bentoml.spacy

        # load a trained pipeline
        nlp = spacy.load("en_core_web_sm")
        # create a doc
        doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
        for token in doc:
            print(token.text, token.pos_, token.dep_)
        ...

        # save the model instance
        tag = bentoml.spacy.save("my_spacy_model", nlp)
        # example tag: my_spacy_model:20211011_52A340

        # load the pipeline back
        nlp = bentoml.spacy.load("my_spacy_model:latest") # or
        nlp = bentoml.spacy.load(tag)
    """  # noqa
    context = {"spacy": spacy.__version__}
    with model_store.register(
        name, module=__name__, framework_context=context, metadata=metadata
    ) as ctx:
        model.to_disk(os.path.join(ctx.path, f"{SAVE_NAMESPACE}.json"))
        return ctx.tag


# class SpacyModel(Model):
#     """
#     Model class for saving/loading :obj:`spacy` models with `spacy.Language.to_disk` and
#     `spacy.util.load_model` methods
#
#     Args:
#         model (`spacy.language.Language`):
#             Every spacy model is of type :obj:`spacy.language.Language`
#         metadata (`GenericDictType`,  `optional`, default to `None`):
#             Class metadata
#
#     Raises:
#         MissingDependencyException:
#             :obj:`spacy` is required by SpacyModel
#
#     Example usage under :code:`train.py`::
#
#         TODO:
#
#     One then can define :code:`bento.py`::
#
#         TODO:
#
#     """
#
#     def __init__(
#         self,
#         model: "spacy.language.Language",
#         metadata: t.Optional[GenericDictType] = None,
#     ):
#         super(SpacyModel, self).__init__(model, metadata=metadata)
#
#     @classmethod
#     def load(cls, path: PathType, **load_model_kwargs) -> "spacy.language.Language":
#         if Path(path).exists():
#             name = os.path.join(path, MODEL_NAMESPACE)
#             model = spacy.util.load_model(name, **load_model_kwargs)
#         else:
#             name = path
#             if name.startswith("blank:"):  # shortcut for blank model
#                 model = spacy.util.get_lang_class(name.replace("blank:", ""))
#             else:
#                 try:  # then we try loading model from the package
#                     spacy.util.load_model_from_package(name, **load_model_kwargs)
#                 except (ModuleNotFoundError, OSError):
#                     logger.warning(
#                         f"module {name} is not available. Installing from pip..."
#                     )
#                     spacy.cli.download(name)
#                 finally:
#                     model = spacy.util.load_model(name, **load_model_kwargs)
#         return model
#
#     def save(self, path: PathType) -> None:
#         self._model.to_disk(os.path.join(path, MODEL_NAMESPACE))
#
#
