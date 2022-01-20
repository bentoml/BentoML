import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import pandas as pd

    from ..models import ModelStore  # noqa


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

MODULE_NAME = "bentoml.h2o"


@inject
def load(
    tag: Tag,
    init_params: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> h2o.model.model_base.ModelBase:
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        init_params (`t.Dict[str, t.Union[str, t.Any]]`):
            Params for h2o server initialization
         model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `h2o.model.model_base.ModelBase`

    Examples:

    .. code-block:: python

        TODO
    """  # noqa

    if not init_params:
        init_params = dict()

    h2o.init(**init_params)

    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )

    path = model.path_of(SAVE_NAMESPACE)
    h2o.no_progress()
    return h2o.load_model(path)


@inject
def save(
    name: str,
    model: h2o.model.model_base.ModelBase,
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`h2o.model.model_base.ModelBase`):
            Instance of h2o model to be saved.
        metadata (`Optional[Dict[str, Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`str`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:
        TODO


    """  # noqa

    context: t.Dict[str, t.Any] = {
        "framework_name": "h2o",
        "pip_dependencies": [f"h2o=={get_pkg_version('h2o')}"],
    }
    options: t.Dict[str, t.Any] = dict()

    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=options,
        context=context,
        metadata=metadata,
    )

    h2o.save_model(model=model, path=_model.path, force=True, filename=SAVE_NAMESPACE)

    _model.save(model_store)

    return _model.tag


class _H2ORunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        predict_fn_name: str,
        init_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]],
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)

        self._tag = Tag.from_taglike(tag)
        self._predict_fn_name = predict_fn_name
        self._init_params = init_params
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[Tag]:
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
    tag: t.Union[str, Tag],
    predict_fn_name: str = "predict",
    *,
    init_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]],
    name: t.Optional[str] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _H2ORunner:
    """Runner represents a unit of serving logic that can be scaled
    horizontally to maximize throughput. `bentoml.h2o.load_runner`
    implements a Runner class that wrap around a h2o model, which
    optimize it for the BentoML runtime.

    Args:
        tag (`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (`str`, default to `predict`):
            Options for inference functions. Default to `predict`
        init_params (`t.Dict[str, t.Union[str, t.Any]]`, default to `None`):
            Parameters for h2o.init(). Refers to https://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html#from-python
             for more information
        resource_quota (`Dict[str, Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`Dict[str, Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.h2o`

    Examples:

    .. code-block:: python

        TODO

    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _H2ORunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        init_params=init_params,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
