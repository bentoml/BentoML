import typing as t
import logging
from typing import TYPE_CHECKING

import attr
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import PathType
from ..models import Model
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

_pycaret_version = get_pkg_version("pycaret")

logger = logging.getLogger(__name__)


def _get_model_info(
    tag: t.Union[str, Tag], model_store: "ModelStore"
) -> t.Tuple["Model", PathType, PathType]:
    model = model_store.get(tag)
    if model.info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with module {model.info.module}, failed loading "
            f"with {__name__}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}")
    pycaret_config = model.path_of(f"{PYCARET_CONFIG}{PKL_EXT}")

    return model, model_file, pycaret_config


@inject
def load(
    tag: t.Union[str, Tag],
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
) -> Tag:
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
    context: t.Dict[str, t.Any] = {
        "framework_name": "pycaret",
        "pip_dependencies": [f"pycaret=={_pycaret_version}"],
    }
    _model = Model.create(
        name,
        module=__name__,
        metadata=metadata,
        context=context,
    )
    save_model(model, _model.path_of(SAVE_NAMESPACE))
    save_config(_model.path_of(f"{PYCARET_CONFIG}{PKL_EXT}"))

    _model.save(model_store)
    return _model.tag


class PycaretRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        self._model_store = model_store
        self._model_tag = Tag.from_taglike(tag)
        name = f"{self.__class__.__name__}_{self._model_tag.name}"
        super().__init__(name, resource_quota, batch_options)

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        _, model_file, pycaret_config = _get_model_info(
            self._model_tag, self._model_store
        )
        load_config(pycaret_config)
        self._model = load_model(model_file)

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
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "PycaretRunner":
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
    return PycaretRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
