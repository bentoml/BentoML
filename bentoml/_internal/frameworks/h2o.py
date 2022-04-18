import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..models import ModelStore


try:
    import h2o  # type: ignore
    import h2o.model  # type: ignore
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
    tag: t.Union[str, Tag],
    init_params: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> h2o.model.model_base.ModelBase:
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        init_params (:code:`Dict[str, Union[str, Any]]`, `optional`, defaults to `None`):
            Params for h2o server initialization
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`h2o.model.model_base.ModelBase`: an instance of `h2o.model.model_base.ModelBase` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        model = bentoml.h2o.load(tag, init_params=dict(port=54323))
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


def save(
    name: str,
    model: h2o.model.model_base.ModelBase,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`h2o.model.model_base.ModelBase`):
            Instance of h2o model to be saved.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml
        import h2o
        import h2o.model
        import h2o.automl

        H2O_PORT = 54323

        def train_h2o_aml() -> h2o.automl.H2OAutoML:

            h2o.init(port=H2O_PORT)
            h2o.no_progress()

            df = h2o.import_file(
                "https://github.com/yubozhao/bentoml-h2o-data-for-testing/raw/master/"
                "powerplant_output.csv"
            )
            splits = df.split_frame(ratios=[0.8], seed=1)
            train = splits[0]
            test = splits[1]

            aml = h2o.automl.H2OAutoML(
                max_runtime_secs=60, seed=1, project_name="powerplant_lb_frame"
            )
            aml.train(y="HourlyEnergyOutputMW", training_frame=train, leaderboard_frame=test)

            return aml

        model = train_h2o_aml()

        tag = bentoml.h2o.save("h2o_model", model.leader)

    """  # noqa

    context: t.Dict[str, t.Any] = {
        "framework_name": "h2o",
        "pip_dependencies": [f"h2o=={get_pkg_version('h2o')}"],
    }
    options: t.Dict[str, t.Any] = dict()

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        options=options,
        context=context,
        metadata=metadata,
    ) as _model:
        h2o.save_model(
            model=model, path=_model.path, force=True, filename=SAVE_NAMESPACE
        )

        return _model.tag


class _H2ORunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        init_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]],
        name: t.Optional[str] = None,
    ):
        super().__init__(tag, name=name)

        self._predict_fn_name = predict_fn_name
        self._init_params = init_params

    @property
    def num_replica(self) -> int:
        return 1

    def _setup(self) -> None:
        _init_params = self._init_params or dict()
        _init_params["nthreads"] = max(round(self.resource_quota.cpu), 1)
        self._model = load(
            self._tag, init_params=_init_params, model_store=self.model_store
        )
        self._predict_fn = getattr(self._model, self._predict_fn_name)  # type: ignore

    def _run_batch(  # type: ignore
        self,
        input_data: t.Union[
            "ext.NpNDArray",
            "ext.PdDataFrame",
            h2o.H2OFrame,
        ],
    ) -> "ext.NpNDArray":
        if not isinstance(input_data, h2o.H2OFrame):
            input_data = h2o.H2OFrame(input_data)
        res = self._predict_fn(input_data)

        if isinstance(res, h2o.H2OFrame):
            res = res.as_data_frame()  # type: ignore
        return np.asarray(res)  # type: ignore


@inject
def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "predict",
    *,
    init_params: t.Optional[t.Dict[str, t.Union[str, t.Any]]],
    name: t.Optional[str] = None,
) -> _H2ORunner:
    """Runner represents a unit of serving logic that can be scaled
    horizontally to maximize throughput. `bentoml.h2o.load_runner`
    implements a Runner class that wrap around a h2o model, which
    optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`predict`):
            Options for inference functions. Default to `predict`
        init_params (:code:`Dict[str, Union[str, Any]]`, default to :code:`None`):
            Parameters for h2o.init(). Refers to `H2O Python API <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html#from-python>`_
            for more information

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.h2o`

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.h2o.load_runner("h2o_model")
        runner.run_batch(data)

    """
    return _H2ORunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        init_params=init_params,
        name=name,
    )
