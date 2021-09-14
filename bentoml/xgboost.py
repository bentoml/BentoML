import os
import typing as t

import bentoml._internal.constants as _const
import bentoml._internal.models.store as _stores

from ._internal.models import (
    JSON_EXT,
    LOAD_INIT_DOCS,
    SAVE_INIT_DOCS,
    SAVE_NAMESPACE,
    SAVE_RETURNS_DOCS,
)
from ._internal.service.runner import Runner
from ._internal.types import GenericDictType
from ._internal.utils import LazyLoader, init_docstrings, returns_docstrings

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import numpy as np
    import pandas as pd
    import xgboost as xgb
else:
    _exc = _const.IMPORT_ERROR_MSG.format(
        fwr="xgboost",
        module=__name__,
        inst="`pip install xgboost`. Refers to"
        " https://xgboost.readthedocs.io/en/latest/install.html"
        " for GPU information.",
    )
    xgb = LazyLoader("xgb", globals(), "xgboost", exc_msg=_exc)
    np = LazyLoader(
        "np", globals(), "numpy", exc_msg="Install numpy with `pip install numpy`"
    )
    pd = LazyLoader(
        "pd", globals(), "pandas", exc_msg="Install pandas with `pip install pandas`"
    )

LOAD_RETURNS_DOCS = """\
    Returns:
        an instance of `xgboost.core.Booster` from BentoML modelstore.
"""


@init_docstrings(LOAD_INIT_DOCS)
@returns_docstrings(LOAD_RETURNS_DOCS)
def load(
    name: str,
    infer_params: t.Dict[str, t.Union[str, int]] = None,
    nthread: int = -1,
) -> "xgb.core.Booster":
    """
    infer_params (`t.Dict[str, t.Union[str, int]]`):
        Params for booster initialization
    nthread (`int`, default to -1):
        Number of thread will be used for this booster.
         Default to -1, which will use XgBoost internal threading
         strategy.
    """
    model_info = _stores.get(name)
    if infer_params is None:
        infer_params = model_info.options
    if "nthread" not in infer_params:
        infer_params["nthread"] = nthread
    return xgb.core.Booster(
        params=infer_params,
        model_file=os.path.join(model_info.path, f"{SAVE_NAMESPACE}{JSON_EXT}"),
    )


@init_docstrings(SAVE_INIT_DOCS)
@returns_docstrings(SAVE_RETURNS_DOCS)
def save(
    name: str,
    model: "xgb.core.Booster",
    *,
    infer_params: t.Dict[str, t.Union[str, int]] = None,
    metadata: t.Optional[GenericDictType] = None,
) -> str:
    """
    model (`xgboost.core.Booster`):
        instance of model to be saved
    """
    with _stores.register(
        name, module=__name__, options=infer_params, metadata=metadata
    ) as ctx:
        model.save_model(os.path.join(ctx.path, f"{SAVE_NAMESPACE}{JSON_EXT}"))
    return f"{name}:{ctx.version}"


def load_runner(*args, **kwargs) -> "_XgBoostRunner":
    return _XgBoostRunner(*args, **kwargs)


class _XgBoostRunner(Runner):
    def __init__(self, name, model_path, *, infer_api_callback: str = "predict"):
        super(_XgBoostRunner, self).__init__(name)
        self._model_path = model_path
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
        # will accept np.ndarray, pd.DataFrame
        infer_params = {"predictor": "cpu_predictor"}

        if self.resource_limits.on_gpu:
            # will accept cupy.ndarray, cudf.DataFrame
            infer_params["predictor"] = "gpu_predictor"
            infer_params["tree_method"] = "gpu_hist"
            # infer_params['gpu_id'] = bentoml.get_gpu_device()

        return infer_params

    def _setup(self) -> None:
        if not self.resource_limits.on_gpu:
            self._model = load(
                self._model_path,
                infer_params=self._infer_params,
                nthread=self.num_concurrency,
            )
        else:
            self._model = load(
                self._model_path, infer_params=self._infer_params, nthread=1
            )
        self._infer_func = getattr(self._model, self._infer_api_callback)

    def _run_batch(
        self, input_data: t.Union["np.ndarray", "pd.DataFrame", "xgb.DMatrix"]
    ) -> "np.ndarray":
        if not isinstance(input_data, xgb.DMatrix):
            input_data = xgb.DMatrix(input_data, nthread=self._num_threads_per_process)
        res = self._infer_func(input_data)
        return np.asarray(res)
