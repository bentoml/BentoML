import os
import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import BentoMLException, MissingDependencyException

if TYPE_CHECKING:
    import pandas as pd

    from ._internal.models.store import ModelInfo, ModelStore, StoreCtx  # noqa


try:
    import h2o
    import h2o.model
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """h2o is required in order to use module `bentoml.h2o`, install h2o
        with `pip install h2o`. For more information, refers to
        https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html#install-in-python
        """
    )


@inject
def load(
    tag: str,
    init_params: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> h2o.model.model_base.ModelBase:
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        init_params (`t.Dict[str, t.Union[str, t.Any]]`):
            Params for h2o server initialization
         model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `h2o.model.model_base.ModelBase`

    Examples::
        TODO
    """  # noqa

    if not init_params:
        init_params = dict()

    h2o.init(**init_params)

    model_info = model_store.get(tag)
    if model_info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with"
            f" module {model_info.module},"
            f" failed loading with {__name__}."
        )

    path = os.path.join(model_info.path, SAVE_NAMESPACE)
    h2o.no_progress()
    return h2o.load_model(path)


@inject
def save(
    name: str,
    model: h2o.model.model_base.ModelBase,
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`h2o.model.model_base.ModelBase`):
            Instance of h2o model to be saved.
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples:
        TODO


    """  # noqa

    context = {"h2o": h2o.__version__}
    options = dict()

    with model_store.register(
        name,
        module=__name__,
        options=options,
        framework_context=context,
        metadata=metadata,
    ) as ctx:  # type: StoreCtx

        h2o.save_model(
            model=model, path=str(ctx.path), force=True, filename=SAVE_NAMESPACE
        )
        return ctx.tag


class _H2ORunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        predict_fn_name: str,
        init_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)

        self._tag = tag
        self._predict_fn_name = predict_fn_name
        self._init_params = init_params
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[str]:
        return [self._tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        nthreads = int(self._init_params.get("nthreads", -1))

        if nthreads == -1:
            return int(round(self.resource_quota.cpu))
        return nthreads

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = load(
            self._tag, init_params=self._init_params, model_store=self._model_store
        )
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self, input_data: t.Union[np.ndarray, "pd.DataFrame", h2o.H2OFrame]
    ) -> np.ndarray:
        if not isinstance(input_data, h2o.H2OFrame):
            input_data = h2o.H2OFrame(input_data)
        res = self._predict_fn(input_data)

        if isinstance(res, h2o.H2OFrame):
            res = res.as_data_frame()
        return np.asarray(res)


@inject
def load_runner(
    tag: str,
    predict_fn_name: str = "predict",
    *,
    init_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]],
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _H2ORunner:
    """Runner represents a unit of serving logic that can be scaled
    horizontally to maximize throughput. `bentoml.h2o.load_runner`
    implements a Runner class that wrap around a h2o model, which
    optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `predict`):
            Options for inference functions. Default to `predict`
        init_params (`t.Dict[str, t.Union[str, t.Any]]`, default to `None`):
            Parameters for h2o.init(). Refers to https://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html#from-python
             for more information
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.h2o`

    Examples::
        TODO

    """  # noqa
    return _H2ORunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        init_params=init_params,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
