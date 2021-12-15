import os
import shutil
import typing as t
from typing import TYPE_CHECKING

import attr
import numpy as np
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml._internal.frameworks import ModelRunner

from ..models import Model
from ..models import PTH_EXT
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from . import AnyNdarray
    from ..models import ModelStore

try:
    import easyocr  # type: ignore

    assert easyocr.__version__ >= "1.4", BentoMLException(
        "Only easyocr>=1.4 is supported by BentoML"
    )

except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """easyocr is required in order to use module `bentoml.easyocr`, install
        easyocr with `pip install easyocr`. For more information, refers to
        https://www.jaided.ai/easyocr/install/
        """
    )

_easyocr_version = get_pkg_version("easyocr")


@inject
def load(
    tag: Tag,
    gpu: bool = True,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> easyocr.Reader:
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        gpu (`bool`):
            Enable GPU support, default to True
        model_store (`~bentoml._internal.models.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `catboost.core.CatBoostClassifier` or `catboost.core. CatBoostRegressor`
        from BentoML modelstore.

    Examples::
        import bentoml.catboost
        booster = bentoml.catboost.load(
            "my_model:20201012_DE43A2", model_params=dict(model_type="classifier"))
    """  # noqa

    model = model_store.get(tag)
    if model.info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with"
            f" module {model.info.module},"
            f" failed loading with {__name__}."
        )

    return easyocr.Reader(
        model_storage_directory=model.path,
        download_enabled=False,
        gpu=gpu,
        **model.info.options,
    )


@inject
def save(
    name: str,
    model: easyocr.Reader,
    *,
    lang_list: t.Optional[t.List[str]] = None,
    recog_network: t.Optional[str] = "english_g2",
    detect_model: t.Optional[str] = "craft_mlt_25k",
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`easyocr.Reader`):
            Instance of model to be saved
        lang_list (`List[str]`, `optional`, default to `["en"]`):
            should be same as `easyocr.Reader(lang_list=...)`
        recog_network (`str`, `optional`, default to `english_g2`):
            should be same as `easyocr.Reader(recog_network=...)`
        detect_model (`str`, `optional`, default to `craft_mlt_25k`):
            TODO:
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        import easyocr

        language_list = ["ch_sim", "en"]
        recog_network = "zh_sim_g2"

        model = easyocr.Reader(
            lang_list=language_list,
            download_enabled=True,
            recog_network=recog_network,
        )

        tag = bentoml.easyocr.save(
            "my_easyocr_model",
            reader,
            lang_list=language_list,
            recog_network=recog_network
        )

        # example tag: my_easyocr_model:20211018_B9EABF

        # load the booster back:
        loaded = bentoml.easyocr.load("my_easyocr_model:latest") # or
        loaded = bentoml.easyocr.load(tag)

    """  # noqa

    context: t.Dict[str, t.Any] = {
        "framework_name": "easyocr",
        "pip_dependencies": [f"easyocr=={_easyocr_version}"],
    }
    if lang_list is None:
        lang_list = ["en"]
    options: t.Dict[str, t.Any] = dict(
        lang_list=lang_list,
        recog_network=recog_network,
    )

    _model = Model.create(
        name,
        module=__name__,
        options=options,
        context=context,
        metadata=metadata,
    )

    src_folder: str = model.model_storage_directory

    detect_filename: str = f"{detect_model}{PTH_EXT}"
    if not os.path.exists(_model.path_of(detect_filename)):
        shutil.copyfile(
            os.path.join(src_folder, detect_filename),
            _model.path_of(detect_filename),
        )

        fname: str = f"{recog_network}{PTH_EXT}"
        shutil.copyfile(os.path.join(src_folder, fname), _model.path_of(fname))

    _model.save(model_store)

    return _model.tag


@attr.define(kw_only=True)
class EasyOCRRunner(ModelRunner):
    predict_fn_name: str = attr.ib()
    predict_params: t.Optional[t.Dict[str, t.Any]] = attr.ib()

    _model: easyocr.Reader = attr.ib(init=False)
    _predict_fn: t.Callable[["AnyNdarray"], "AnyNdarray"] = attr.ib(init=False)

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    def _setup(self) -> None:
        self._model = load(
            self.tag,
            self.resource_quota.on_gpu,
            self.model_store,
        )
        self._predict_fn = getattr(self._model, self.predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self, input_data: "AnyNdarray"
    ) -> "AnyNdarray":
        if self.predict_params is None:
            res = self._predict_fn(input_data)
        else:
            res = self._predict_fn(input_data, **self.predict_params)
        return np.asarray(res)  # type: ignore


@inject
def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "readtext_batched",
    *,
    predict_params: t.Union[None, t.Dict[str, t.Union[str, t.Any]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    name: t.Optional[str] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
) -> EasyOCRRunner:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.easyocr.load_runner` implements a Runner class that
    wrap around a EasyOCR Reader model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `readtext_batched`):
            Options for inference functions. Default to `readtext_batched`
        booster_params (`t.Dict[str, t.Union[str, int]]`, default to `None`):
            Parameters for boosters. Refers to https://xgboost.readthedocs.io/en/latest/parameter.html
             for more information
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.ModelStore`, default to `BentoMLContainer.model_store`):
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
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name

    return EasyOCRRunner(  # pylint: disable=unexpected-keyword-arg
        tag=tag,
        predict_fn_name=predict_fn_name,
        predict_params=predict_params,
        model_store=model_store,
        name=name,
        resource_quota=resource_quota,  # type: ignore
        batch_options=batch_options,  # type: ignore
    )
