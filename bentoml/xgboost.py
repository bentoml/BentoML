import os
import typing as t

import bentoml._internal.constants as _const

from ._internal.models.base import JSON_EXTENSION, MODEL_NAMESPACE, Model
from ._internal.service.runner import Runner
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

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


def _save_model(
    name: str,
    model: "xgb.core.Booster",
    *,
    metadata: t.Optional[MetadataType] = None,
    **save_options,
):
    _instance = _XgBoostModel(model, metadata=metadata)


save = _save_model


def _load_model(name: str):
    ...


load = _load_model


class _XgBoostModel(Model):
    """
    Artifact class for saving/loading :obj:`xgboost` model

    Args:
        model (`xgboost.core.Booster`):
            Every xgboost model instance of type :obj:`xgboost.core.Booster`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`xgboost` is required by _XgBoostModel
        TypeError:
           model must be instance of :obj:`xgboost.core.Booster`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:

    """

    def __init__(
        self, model: "xgb.core.Booster", metadata: t.Optional[MetadataType] = None
    ):
        super(_XgBoostModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(  # noqa # pylint: disable=arguments-differ
        cls,
        path: PathType,
        infer_params: t.Dict[str, t.Union[str, int]] = None,
        nthread: int = -1,
    ) -> "xgb.core.Booster":
        if "nthread" not in infer_params and nthread > 0:
            infer_params["nthread"] = nthread
        return xgb.core.Booster(
            params=infer_params,
            model_file=os.path.join(path, f"{MODEL_NAMESPACE}{JSON_EXTENSION}"),
        )

    def save(self, path: PathType) -> None:
        self._model.save_model(os.path.join(path, f"{MODEL_NAMESPACE}{JSON_EXTENSION}"))

    @classmethod
    def load_runner(cls, *runner_args, **runner_kwargs):
        return _XgBoostRunner(*runner_args, **runner_kwargs)


class _XgBoostRunner(Runner):
    @property
    def num_concurrency(self):
        if self._on_gpu:
            return 1
        return self._num_threads_per_process

    @property
    def num_replica(self):
        if self._on_gpu:
            return self.GPU
        return 1

    @property
    def _num_threads_per_process(self):
        return int(round(self.CPU))

    def __init__(
        self,
        model_path,
        infer_api_callback: str = "predict",
        on_gpu=False,
        batch_axis=0,
    ):
        super(_XgBoostRunner, self).__init__(
            model_path, infer_api_callback=infer_api_callback
        )
        self._batch_axis = batch_axis
        self._on_gpu = on_gpu
        self._infer_api_callback = "predict"
        # information to pass to  marshal server
        infer_params = {
            "predictor": "cpu_predictor"
        }  # will accept np.ndarray, pd.DataFrame

        if self._on_gpu:
            infer_params[
                "predictor"
            ] = "gpu_predictor"  # will accept cupy.ndarray, cudf.DataFrame
            infer_params["tree_method"] = "gpu_hist"

            # infer_params['gpu_id'] = bentoml.get_gpu_device()
        self._infer_params = infer_params

    def _setup(self) -> None:
        if not self._on_gpu:
            self._model = _XgBoostModel.load(
                self._model_path,
                infer_params=self._infer_params,
                nthread=self.num_concurrency,
            )
        else:
            self._model = _XgBoostModel.load(
                self._model_path, infer_params=self._infer_params, nthread=1
            )
        self._infer_func = getattr(self._model, self._infer_api_callback)

    def _run_batch(
        self, input_data: t.Union["np.ndarray", "pd.DataFrame", "xgb.DMatrix"]
    ) -> "np.ndarray":
        if not isinstance(input_data, xgb.DMatrix):
            input_data = xgb.DMatrix(input_data, nthreads=self._num_threads_per_process)
        res = self._infer_func(input_data)
        return np.asarray(res)
