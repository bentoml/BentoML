import logging
import os
import typing as t

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import PKL_EXT, SAVE_NAMESPACE
from ._internal.runner import Runner
from ._internal.types import PathType
from .exceptions import BentoMLException, MissingDependencyException

PYCARET_CONFIG = "pycaret_config"

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import lightgbm
    import pandas as pd
    import sklearn
    import xgboost
    from _internal.models.store import ModelInfo, ModelStore

try:
    from pycaret.internal.tabular import (
        load_config,
        load_model,
        predict_model,
        save_config,
        save_model,
    )
    from pycaret.utils import version
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        pycaret is required in order to use module`bentoml.pycaret`, install
         pycaret with `pip install pycaret` or `pip install pycaret[full]`.
         For more information, refer to
         https://pycaret.readthedocs.io/en/latest/installation.html
        """
    )

logger = logging.getLogger(__name__)


def _get_model_info(
    tag: str, model_store: "ModelStore"
) -> t.Tuple["ModelInfo", PathType, PathType]:
    model_info = model_store.get(tag)
    if model_info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with module {model_info.module}, failed loading "
            f"with {__name__}."
        )
    model_file = os.path.join(model_info.path, f"{SAVE_NAMESPACE}")
    pycaret_config = os.path.join(model_info.path, f"{PYCARET_CONFIG}{PKL_EXT}")

    return model_info, model_file, pycaret_config


@inject
def load(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Any:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        a Pycaret model instance. This depends on which type of model you save with BentoML (Classifier, Trees, SVM, etc).

    Examples:
        import bentoml.pycaret
        # NOTE: pycaret setup config will be loaded
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
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Union[sklearn.pipeline.Pipeline, xgboost.Booster, lightgbm.basic.Booster]`):
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples:
        from pycaret.classification import *
        import bentoml.pycaret
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
    context = {"pycaret": version()}
    with model_store.register(
        name,
        module=__name__,
        metadata=metadata,
        framework_context=context,
    ) as ctx:
        save_model(model, os.path.join(ctx.path, SAVE_NAMESPACE))
        save_config(os.path.join(ctx.path, f"{PYCARET_CONFIG}{PKL_EXT}"))
        return ctx.tag  # type: ignore[no-any-return]


class _PycaretRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        model_info, model_file, pycaret_config = _get_model_info(tag, model_store)

        self._model_info = model_info
        self._model_file = model_file
        self._pycaret_config = pycaret_config

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

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
        return predict_model(self._model, input_data)


@inject
def load_runner(
    tag: str,
    *,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PycaretRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.pycaret.load_runner` implements a Runner class that
    wraps around a PyCaret model, which optimizes it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.xgboost` model

    Examples:
        import bentoml.pycaret
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

        # set up the runner
        runner._setup()

        prediction = runner._run_batch(input_data=data_unseen)
        print(prediction)
    """  # noqa
    return _PycaretRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
