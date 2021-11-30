import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject, Provide

from .exceptions import BentoMLException, MissingDependencyException
from ._internal.types import Tag, PathType
from ._internal.models import Model, PKL_EXT, SAVE_NAMESPACE
from ._internal.runner import Runner
from ._internal.configuration.containers import BentoMLContainer

_MT = t.TypeVar("_MT")

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ._internal.models import ModelStore

try:
    import joblib
    import sklearn
    from joblib import parallel_backend

except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """sklearn is required in order to use the module `bentoml.sklearn`, install
         sklearn with `pip install sklearn`. For more information, refer to
         https://scikit-learn.org/stable/install.html
        """
    )


def _get_model_info(
    tag: t.Union[str, Tag], model_store: "ModelStore"
) -> t.Tuple["Model", PathType]:
    model = model_store.get(tag)
    if model.info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with module {model.info.module}, failed loading"
            f" with {__name__}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    return model, model_file


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _MT:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of sklearn model from BentoML modelstore.

    Examples:
        import bentoml.sklearn
        sklearn = bentoml.sklearn.load('my_model:20201012_DE43A2')

    """  # noqa
    _, model_file = _get_model_info(tag, model_store)
    _load: t.Callable[[PathType], _MT] = joblib.load
    return _load(model_file)


@inject
def save(
    name: str,
    model: _MT,
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (``):
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples:

    """  # noqa
    context = {"sklearn": sklearn.__version__}

    _model = Model.create(
        name,
        module=__name__,
        metadata=metadata,
        framework_context=context,
    )

    joblib.dump(model, _model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"))

    _model.save(model_store)
    return _model.tag


class _SklearnRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        function_name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(str(tag), resource_quota, batch_options)
        model_info, model_file = _get_model_info(tag, model_store)
        self._model_store = model_store
        self._model_info = model_info
        self._model_file = model_file
        self._parallel_ctx = parallel_backend(
            "threading", n_jobs=self.num_concurrency_per_replica
        )
        self._function_name = function_name

    @property
    def num_concurrency_per_replica(self) -> int:
        # NOTE: sklearn doesn't use GPU, so return max. no. of CPU's.
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        # NOTE: sklearn doesn't use GPU, so just return 1.
        return 1

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = joblib.load(filename=self._model_file)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self,
        *args: t.Union[np.ndarray, "pd.DataFrame"],
        **kwargs: t.Union[np.ndarray, "pd.DataFrame"],
    ) -> "np.ndarray":
        func = getattr(self._model, self._function_name)
        with self._parallel_ctx:
            return func(*args, **kwargs)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    function_name: str = "predict",
    *,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_SklearnRunner":

    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.sklearn.load_runner` implements a Runner class that
    wrap around a Sklearn joblib model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore.
        function_name (`str`, `optional`, default to `predict`):
            Predict function used by a given sklearn model.
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for the target `bentoml.sklearn` model

    Examples::
        import bentoml
        import bentoml.sklearn
        import numpy as np

        from bentoml.io import NumpyNdarray

        input_data = NumpyNdarray()
        runner = bentoml.sklearn.load_runner("my_model:20201012_DE43A2")
        runner.run(input_data)
    """  # noqa
    return _SklearnRunner(
        tag=tag,
        function_name=function_name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
