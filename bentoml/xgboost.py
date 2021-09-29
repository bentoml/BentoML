import os
import typing as t

from ._internal import constants as _const
from ._internal.models import JSON_EXT, SAVE_NAMESPACE
from ._internal.models import store as _stores
from ._internal.service import Runner
from ._internal.types import GenericDictType
from .exceptions import MissingDependencyException
from .utils import docstrings  # noqa

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="xgboost",
    module=__name__,
    inst="`pip install xgboost`. Refers to"
    " https://xgboost.readthedocs.io/en/latest/install.html"
    " for GPU information.",
)
try:
    import numpy as np
    import pandas as pd
    import xgboost as xgb
except ImportError:
    raise MissingDependencyException(_exc)


@docstrings(
    """\
Load a model from BentoML modelstore with given name.

Args:
    name (`str`):
        Name of a saved model in BentoML modelstore.
    infer_params (`t.Dict[str, t.Union[str, int]]`):
        Params for booster initialization
    nthread (`int`, default to -1):
        Number of thread will be used for this booster.
         Default to -1, which will use XgBoost internal threading
         strategy.

Returns:
    an instance of `xgboost.core.Booster` from BentoML modelstore.

Examples::
    import bentoml.xgboost
    model = bentoml.xgboost.load('xgboost_ngrams', infer_params=dict(gpu_id=0))
"""
)
def load(
    name: str,
    infer_params: t.Dict[str, t.Union[str, int]] = None,
    nthread: int = -1,
    **xgboost_load_args,
) -> "xgb.core.Booster":
    model_info = _stores.get(name)
    if infer_params is None:
        infer_params = model_info.options
    if "nthread" not in infer_params:
        infer_params["nthread"] = nthread
    return xgb.core.Booster(
        params=infer_params,
        model_file=os.path.join(model_info.path, f"{SAVE_NAMESPACE}{JSON_EXT}"),
        **xgboost_load_args,
    )


@docstrings(
    """\
Save a model instance to BentoML modelstore.

Args:
    name (`str`):
        Name for given model instance. This should pass Python identifier check.
    metadata (`~bentoml._internal.types.GenericDictType`, default to `None`):
        Custom metadata for given model.
    model (`xgboost.core.Booster`):
        Instance of model to be saved
    infer_params (`t.Dict[str, t.Union[str, int]]`):
        Params for booster initialization

Returns:
    store_name (`str` with a format `name:generated_id`) where `name` is the defined name
    user set for their models, and `generated_id` will be generated UUID by BentoML.

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

    tag = bentoml.xgboost.save("xgboost_tree", bst, infer_params = param)
"""
)
def save(
    name: str,
    model: "xgb.core.Booster",
    *,
    infer_params: t.Dict[str, t.Union[str, int]] = None,
    metadata: t.Optional[GenericDictType] = None,
) -> str:
    context = {"xgboost": xgb.__version__}
    with _stores.register(
        name,
        module=__name__,
        options=infer_params,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        model.save_model(os.path.join(ctx.path, f"{SAVE_NAMESPACE}{JSON_EXT}"))
    return f"{name}:{ctx.version}"


@docstrings(
    """\
Runner is a tiny computation unit that runs framework inference. Runner makes it
easy to leverage multiple threads or processes in a Python-centric model serving
architecture, where users can also have the ability to fine tune the performance
per thread/processes level.

Usage::
    r = bentoml.xgboost.load_runner()
    r.resource_limits.cpu = 2
    r.resource_limits.mem = "2Gi"

    # Runners config override:
    runners = {
        "my_model:any": {
            "resource_limits": {"cpu": 1},
            "batch_options": {"max_batch_size": 1000},
        },
        "runner_bar": {"resource_limits": {"cpu": "200m"}},
    }

    # bentoml.xgboost.py:
    class _XgboostRunner(Runner):

        def __init__(self, runner_name, model_path):
            super().__init__(name)

        def _setup(self):
            self.model = load(model_path)
            ...

    # model_tag example:
    #  "my_nlp_model:20210810_A23CDE", "my_nlp_model:latest"
    def load_runner(model_tag: str):
        model_info = bentoml.models.get(model_tag)
        assert model_info.module == "bentoml.xgboost"
        return _XgboostRunner(model_tag, model_info.path)

    def save(name: str, model: xgboost.Model, **save_options):
        with bentoml.models.add(
            name,
            module=__module__,
            options: save_options) as ctx:

        # ctx( path, version, metadata )
        model.save(ctx.path)
        ctx.metadata.set('param_a', 'value_b')
        ctx.metadata.set('param_foo', 'value_bar')

    def load(name: str) -> xgboost.Model:
        model_info = bentoml.models.get(model_tag)
        assert model_info.module == "bentoml.xgboost"
        return xgboost.load_model(model_info.path)

        # custom runner
    class _MyRunner(Runner):

        def _setup(self):
            self.model = load("./my_model.pt")

        def _run_batch(self, ...):
            pass

Users can also customize options for each Runner with::
    cpu (`float`, default to `1.0`): # of CPUs for a Runner
    mem (`Union[str, int]`, default to `100Mi`): Default memory allocated for a Runner
    gpu (`float`, default to `0.0`): # of GPUs for a Runner
    enable_batch (`bool`, default to `True`): enable dynamic-batching by default
    max_batch_size (`int`, default to `10000`): maximum batch size
    max_latency_ms (`int`, default to `10000`): maximum latency before raises Error

Args:
    name (`str`):
        Model name previously saved in BentoML modelstore.
    runner_name (`Optional[str]`, default to `module_name_uuid`):
        name of given runner
    infer_api_callback (`str`, default to `predict`):
        callback function for inference call for `xgboost.core.Booster`

Returns:
    Runner instances for `bentoml.xgboost`. Users can then access `run_batch` to start running inference.


Examples::
    import xgboost as xgb
    import bentoml.xgboost
    import pandas as pd
    
    input_data = pd.from_csv("/path/to/csv")
    
    runner = bentoml.xgboost.load_runner("/path/to/model")
    runner.run_batch(xgb.DMatrix(input_data))
"""
)
def load_runner(
    name: str, model_path: str, infer_api_callback: str = "predict"
) -> "_XgBoostRunner":
    return _XgBoostRunner(
        name=name, model_path=model_path, infer_api_callback=infer_api_callback
    )


class _XgBoostRunner(Runner):
    def __init__(
        self,
        name: str,
        model_path: str,
        infer_api_callback: str = "predict",
    ):
        super(_XgBoostRunner, self).__init__(name, model_path)
        self._infer_api_callback = infer_api_callback
        self._infer_params = self._setup_infer_params()

    @property
    def num_concurrency(self):
        if self.resource_limits.on_gpu:
            return 1
        return self._num_threads_per_process

    @property
    def num_replica(self):
        if self.resource_limits.on_gpu:
            return self.resource_limits.gpu
        return 1

    @property
    def _num_threads_per_process(self):
        return int(round(self.resource_limits.cpu))

    def _setup_infer_params(self):
        infer_params = {"predictor": "cpu_predictor"}

        if self.resource_limits.on_gpu:
            infer_params["predictor"] = "gpu_predictor"
            infer_params["tree_method"] = "gpu_hist"
            # TODO: bentoml.get_gpu_device()
            # infer_params['gpu_id'] = bentoml.get_gpu_device()

        return infer_params

    def _setup(self) -> None:
        if not self.resource_limits.on_gpu:
            self._model = load(
                self.model_path,
                infer_params=self._infer_params,
                nthread=self.num_concurrency,
            )
        else:
            self._model = load(
                self.model_path, infer_params=self._infer_params, nthread=1
            )
        self._infer_func = getattr(self._model, self._infer_api_callback)

    def _run_batch(
        self, input_data: t.Union["np.ndarray", "pd.DataFrame", "xgb.DMatrix"]
    ) -> "np.ndarray":
        if not isinstance(input_data, xgb.DMatrix):
            input_data = xgb.DMatrix(input_data, nthread=self._num_threads_per_process)
        res = self._infer_func(input_data)
        return np.asarray(res)
