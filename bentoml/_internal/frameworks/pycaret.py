import typing as t
import logging
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import PathType
from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

PYCARET_CONFIG = "pycaret_config"

if TYPE_CHECKING:
    import pandas as pd
    import sklearn
    import xgboost
    import lightgbm

    from ..models import ModelStore

try:
    from pycaret.internal.tabular import load_model
    from pycaret.internal.tabular import save_model
    from pycaret.internal.tabular import load_config
    from pycaret.internal.tabular import save_config
    from pycaret.internal.tabular import predict_model
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        pycaret is required in order to use module`bentoml.pycaret`, install
         pycaret with `pip install pycaret` or `pip install pycaret[full]`.
         For more information, refer to
         https://pycaret.readthedocs.io/en/latest/installation.html
        """
    )

MODULE_NAME = "bentoml.pycaret"

logger = logging.getLogger(__name__)


def _get_model_info(
    tag: Tag, model_store: "ModelStore"
) -> t.Tuple["Model", PathType, PathType]:
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}")
    pycaret_config = model.path_of(f"{PYCARET_CONFIG}{PKL_EXT}")

    return model, model_file, pycaret_config


@inject
def load(
    tag: Tag,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Any:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`Any`: a :mod:`Pycaret` model instance.

    Examples:

    .. code-block:: python

        import bentoml
        dt = bentoml.pycaret.load("my_model:latest")
    """  # noqa
    _, model_file, pycaret_config = _get_model_info(tag, model_store)
    load_config(pycaret_config)
    return load_model(model_file)


@inject
def save(
    name: str,
    model: t.Union[
        "sklearn.pipeline.Pipeline", "xgboost.Booster", "lightgbm.basic.Booster"
    ],
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[sklearn.pipeline.Pipeline, xgboost.Booster, lightgbm.basic.Booster]`):
            Instance of model to be saved
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        from pycaret.classification import *
        from pycaret.datasets import get_data

        # get data
        dataset = get_data('credit')
        data = dataset.sample(frac=0.95, random_state=786)

        # setup is needed to save
        exp_clf101 = setup(data = data, target = 'default', session_id=123)

        best_model = compare_models() # get the best model
        tuned_best = tune_model(best_model)
        final_model = finalize_model(tuned_best)

        # NOTE: pycaret setup config will be saved with the model
        bentoml.pycaret.save("my_model", final_model)
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "pycaret",
        "pip_dependencies": [f"pycaret=={get_pkg_version('pycaret')}"],
    }
    _model = Model.create(
        name,
        module=MODULE_NAME,
        metadata=metadata,
        context=context,
    )
    save_model(model, _model.path_of(SAVE_NAMESPACE))
    save_config(_model.path_of(f"{PYCARET_CONFIG}{PKL_EXT}"))

    _model.save(model_store)
    return _model.tag


class _PycaretRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)
        model_info, model_file, pycaret_config = _get_model_info(tag, model_store)

        self._model_info = model_info
        self._model_file = model_file
        self._pycaret_config = pycaret_config

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        load_config(self._pycaret_config)
        self._model = load_model(self._model_file)

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[override] # noqa # pylint: disable
        logger.warning(
            "PyCaret is not designed to be ran"
            " in parallel. See https://github.com/pycaret/pycaret/issues/758"  # noqa # pylint: disable
        )
        output = predict_model(self._model, input_data)  # type: pd.DataFrame
        return output


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    name: t.Optional[str] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PycaretRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.pycaret.load_runner` implements a Runner class that
    wraps around a PyCaret model, which optimizes it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.pycaret` model

    Examples:

    .. code-block:: python

        import bentoml
        from pycaret.datasets import get_data
        dataset = get_data('credit')

        # pre-process data
        data = dataset.sample(frac=0.95, random_state=786)
        data_unseen = dataset.drop(data.index)
        data.reset_index(inplace=True, drop=True)
        data_unseen.reset_index(inplace=True, drop=True)

        # initialize runner
        # NOTE: loading the model will load the saved pycaret config too
        runner = bentoml.pycaret.load_runner(tag="my_model:latest")

        prediction = runner.run_batch(input_data=data_unseen)
    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _PycaretRunner(
        tag=tag,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
