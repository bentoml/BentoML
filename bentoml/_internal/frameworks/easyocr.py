import os
import shutil
import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PTH_EXT
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..models import ModelStore

try:
    import easyocr
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """easyocr is required in order to use module `bentoml.easyocr`, install
        easyocr with `pip install easyocr`. For more information, refers to
        https://www.jaided.ai/easyocr/install/
        """
    )

MODULE_NAME = "bentoml.easyocr"


@inject
def load(
    tag: str,
    gpu: bool = True,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> easyocr.Reader:
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        gpu (`bool`, `optional`, defaults to True):
            Enable GPU support.
        model_store (`~bentoml._internal.models.ModelStore`, default to :code:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`easyocr.Reader`: an instance of :code:`easyocr.Reader` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        booster = bentoml.easyocr.load("craft_model", model_params=dict(model_type="classifier"))
    """  # noqa
    assert easyocr.__version__ >= "1.4", BentoMLException(
        "Only easyocr>=1.4 is supported by BentoML"
    )

    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
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
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`easyocr.Reader`):
            Instance of model to be saved.
        lang_list (`List[str]`, `optional`, default to :code:`None`):
            Should be same as :func:`easyocr.Reader(lang_list=...)`.
            If `None`, then default to :code:`["en"]`.
        recog_network (:code:`str`, `optional`, default to :code:`english_g2`):
            Should be same as `easyocr.Reader(recog_network=...)`
        detect_model (:code:`str`, `optional`, default to :code:`craft_mlt_25k`):
            Model used for detection pipeline.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.ModelStore`, default to :code:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import easyocr
        import bentoml

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

        # load the booster back:
        loaded = bentoml.easyocr.load("my_easyocr_model:latest") # or
        loaded = bentoml.easyocr.load(tag)

    """  # noqa
    assert easyocr.__version__ >= "1.4", BentoMLException(
        "Only easyocr>=1.4 is supported by BentoML"
    )

    context: t.Dict[str, t.Any] = {
        "framework_name": "easyocr",
        "pip_dependencies": [f"easyocr=={get_pkg_version('easyocr')}"],
    }
    if lang_list is None:
        lang_list = ["en"]
    options: t.Dict[str, t.Any] = dict(
        lang_list=lang_list,
        recog_network=recog_network,
    )

    _model = Model.create(
        name,
        module=MODULE_NAME,
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


class _EasyOCRRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        predict_fn_name: str,
        name: str,
        predict_params: t.Optional[t.Dict[str, t.Any]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)
        self._tag = tag
        self._predict_fn_name = predict_fn_name
        self._predict_params = predict_params
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = load(self._tag, self.resource_quota.on_gpu, self._model_store)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self, input_data: np.ndarray
    ) -> "np.ndarray":
        res = self._predict_fn(input_data, **self._predict_params)
        return np.asarray(res, dtype=object)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "readtext_batched",
    *,
    name: t.Optional[str] = None,
    predict_params: t.Union[None, t.Dict[str, t.Union[str, t.Any]]] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _EasyOCRRunner:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.easyocr.load_runner` implements a Runner class that
    wrap around a EasyOCR Reader model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`readtext_batched`):
            Options for inference functions.
        predict_params (:code:`Dict[str, Union[str, Any]]`, `optional`, default to :code:`None`):
            Parameters for prediction. Refers `here <https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py#L460>`_
            for more information.
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.ModelStore`, default to :code:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.easyocr` model

    Examples:

    .. code-block:: python

        import bentoml
        import pandas as pd

        input_data = pd.from_csv("/path/to/csv")
        runner = bentoml.easyocr.load_runner("my_easyocr_model")
        runner.run(pd.DataFrame(input_data))
    """
    assert easyocr.__version__ >= "1.4", BentoMLException(
        "Only easyocr>=1.4 is supported by BentoML"
    )
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _EasyOCRRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        predict_params=predict_params,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
