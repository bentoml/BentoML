import os
import typing as t

import bentoml._internal.constants as const

from ._internal.models.base import JSON_EXTENSION, MODEL_NAMESPACE, Model
from ._internal.service.runner import _Runner as _Runner
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import xgboost as xgb
else:
    _exc = const.IMPORT_ERROR_MSG.format(
        fwr="xgboost",
        module=__name__,
        inst="`pip install xgboost`. Refers to"
        " https://xgboost.readthedocs.io/en/latest/install.html"
        " for GPU information.",
    )
    xgb = LazyLoader("xgb", globals(), "xgboost", exc_msg=_exc)


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
        cls, path: PathType, infer_params: t.Dict[str, t.Union[str, int]] = None
    ) -> "xgb.core.Booster":
        return xgb.core.Booster(
            params=infer_params,
            model_file=os.path.join(path, f"{MODEL_NAMESPACE}{JSON_EXTENSION}"),
        )

    def save(self, path: PathType) -> None:
        self._model.save_model(os.path.join(path, f"{MODEL_NAMESPACE}{JSON_EXTENSION}"))

    @classmethod
    def load_runner(cls, *runner_args, **runner_kwargs):
        return _XgBoostRunner(*runner_args, **runner_kwargs)


class _XgBoostRunner(_Runner):
    CPU: float = 1.0
    RAM: str = "100M"
    GPU: float = 0.0

    dynamic_batching = True
    max_batch_size = 10000
    max_latency_ms = 10000

    _on_gpu: bool = False

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
