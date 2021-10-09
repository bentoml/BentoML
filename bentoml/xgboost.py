import os
import typing as t

import numpy as np
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import MODEL_EXT, SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import BentoMLException, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import pandas as pd

    from ._internal.models.store import ModelStore

try:
    import xgboost as xgb
except ImportError:
    raise MissingDependencyException(
        """xgboost is required in order to use module `bentoml.xgboost`, install
        xgboost with `pip install xgboost`. For more information, refers to
        https://xgboost.readthedocs.io/en/latest/install.html
        """
    )

# TODO: support xgb.DMatrix runner io container
# from bentoml.runner import RunnerIOContainer, register_io_container
# class DMatrixContainer(RunnerIOConatainer):
#     batch_type = xgb.DMatrix
#     item_type = xgb.DMatrix
#
#     def flatten(self):
#         pass
#
#     def squeeze(self):
#         pass
#
#     def serialize(self):
#         pass
#
#     def deserialize(self):
#         pass
#
# register_io_container(DMatrixContainer)


def _get_model_info(
    tag: str, booster_params: t.Dict[str, t.Any], model_store: "ModelStore"
):
    model_info = model_store.get(tag)
    if model_info.module != __name__:
        raise BentoMLException(
            f"Model {tag} was saved with module {model_info.module}, failed loading "
            f"with {__name__}."
        )
    model_file = os.path.join(model_info.path, f"{SAVE_NAMESPACE}{MODEL_EXT}")
    booster_params = dict() if booster_params is None else booster_params
    for key, value in model_info.options.items():
        if key not in booster_params:
            booster_params[key] = value  # apply booster_params override
    if "nthread" not in booster_params:
        booster_params["nthread"] = -1  # apply default nthread parameter

    return model_info, model_file, booster_params


@inject
def load(
    tag: str,
    booster_params: t.Dict[str, t.Union[str, int]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "xgb.core.Booster":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        booster_params (`t.Dict[str, t.Union[str, int]]`):
            Params for xgb.core.Booster initialization
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `xgboost.core.Booster` from BentoML modelstore.

    Examples::
        import bentoml.xgboost
        booster = bentoml.xgboost.load(
            'my_model:20201012_DE43A2', booster_params=dict(gpu_id=0))
    """  # noqa
    booster_params = dict() if booster_params is None else booster_params
    _, model_file, booster_params = _get_model_info(tag, booster_params, model_store)

    return xgb.core.Booster(
        params=booster_params,
        model_file=model_file,
    )


@inject
def save(
    name: str,
    model: "xgb.core.Booster",
    *,
    booster_params: t.Dict[str, t.Union[str, int]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`xgboost.core.Booster`):
            Instance of model to be saved
        booster_params (`t.Dict[str, t.Union[str, int]]`):
            Params for booster initialization
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        import xgboost as xgb
        import bentoml.xgboost

        # read in data
        dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
        dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
        # specify parameters via map
        param = dict(max_depth=2, eta=1, objective='binary:logistic')
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)
        ...

        tag = bentoml.xgboost.save("my_xgboost_model", bst, booster_params=param)
        # example tag: my_xgboost_model:20210929_153BC4

        # load the booster back:
        bst = bentoml.xgboost.load("my_xgboost_model:latest") # or
        bst = bentoml.xgboost.load(tag)
    """  # noqa
    context = {"xgboost": xgb.__version__}
    with model_store.register(
        name,
        module=__name__,
        options=booster_params,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        model.save_model(os.path.join(ctx.path, f"{SAVE_NAMESPACE}{MODEL_EXT}"))
        return ctx.tag


class _XgBoostRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        predict_fn_name: str,
        booster_params: t.Dict[str, t.Union[str, int]],
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        model_info, model_file, booster_params = _get_model_info(
            tag, booster_params, model_store
        )

        self._model_info = model_info
        self._model_file = model_file
        self._predict_fn_name = predict_fn_name
        self._booster_params = self._setup_booster_params(booster_params)

    @property
    def required_models(self):
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self):
        if self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self):
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    def _setup_booster_params(self, booster_params):
        if self.resource_quota.on_gpu:
            booster_params["predictor"] = "gpu_predictor"
            booster_params["tree_method"] = "gpu_hist"
            # Use the first device reported by CUDA runtime
            booster_params["gpu_id"] = 0
            booster_params["nthread"] = 1
        else:
            booster_params["predictor"] = "cpu_predictor"
            booster_params["nthread"] = self.num_concurrency_per_replica

        return booster_params

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(
        self,
    ) -> None:
        self._model = xgb.core.Booster(
            params=self._booster_params,
            model_file=self._model_file,
        )
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    def _run_batch(  # pylint: disable=arguments-differ
        self, input_data: t.Union[np.ndarray, "pd.DataFrame", xgb.DMatrix]
    ) -> "np.ndarray":
        if not isinstance(input_data, xgb.DMatrix):
            input_data = xgb.DMatrix(input_data)
        res = self._predict_fn(input_data)
        return np.asarray(res)


@inject
def load_runner(
    tag: str,
    predict_fn_name: str = "predict",
    *,
    booster_params: t.Dict[str, t.Union[str, int]] = None,
    resource_quota: t.Dict[str, t.Any] = None,
    batch_options: t.Dict[str, t.Any] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _XgBoostRunner:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.xgboost.load_runner` implements a Runner class that
    wrap around a Xgboost booster model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `predict`):
            Options for inference functions. If you want to use `run`
             or `run_batch` in a thread context then use `inplace_predict`.
             Otherwise, `predict` are the de facto functions.
        booster_params (`t.Dict[str, t.Union[str, int]]`, default to `None`):
            Parameters for boosters. Refers to https://xgboost.readthedocs.io/en/latest/parameter.html
             for more information
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.xgboost` model

    Examples::
        import xgboost as xgb
        import bentoml.xgboost
        import pandas as pd

        input_data = pd.from_csv("/path/to/csv")
        runner = bentoml.xgboost.load_runner("my_model:20201012_DE43A2")
        runner.run(xgb.DMatrix(input_data))
    """  # noqa
    return _XgBoostRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        booster_params=booster_params,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
