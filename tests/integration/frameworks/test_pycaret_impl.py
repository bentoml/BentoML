import typing as t

import pytest
import sklearn
from pycaret.datasets import get_data
from pycaret.classification import setup as pycaret_setup
from pycaret.classification import save_model
from pycaret.classification import tune_model
from pycaret.classification import create_model
from pycaret.classification import predict_model
from pycaret.classification import finalize_model

import bentoml
import bentoml.models
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension

if t.TYPE_CHECKING:
    from bentoml._internal.models import Model
    from bentoml._internal.models import ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


@pytest.fixture()
def get_pycaret_data():
    dataset = get_data("credit")
    data = dataset.sample(frac=0.95, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(inplace=True, drop=True)
    data_unseen.reset_index(inplace=True, drop=True)

    return data, data_unseen[:5]


@pytest.fixture()
def pycaret_model(get_pycaret_data) -> t.Any:
    # note: silent must be set to True to avoid the confirmation input of data types
    train_data, _ = get_pycaret_data
    pycaret_setup(data=train_data, target="default", session_id=123, silent=True)
    dt = create_model("dt")
    tuned_dt = tune_model(dt)
    final_dt = finalize_model(tuned_dt)

    return final_dt


@pytest.fixture()
def save_proc(
    pycaret_model,
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "Model"]:
    def _(metadata) -> "Model":
        tag = bentoml.pycaret.save(
            TEST_MODEL_NAME, pycaret_model, metadata=metadata, model_store=modelstore
        )
        model = modelstore.get(tag)
        return model

    return _


@pytest.fixture()
def wrong_module(pycaret_model, modelstore: "ModelStore"):
    with bentoml.models.create(
        "wrong_module",
        module=__name__,
        options=None,
        framework_context=None,
        metadata=None,
    ) as _model:
        save_model(pycaret_model, _model.path_of("saved_model.pkl"))
        return _model.path


@pytest.mark.parametrize(
    "metadata",
    [
        ({"acc": 0.876}),
    ],
)
def test_pycaret_save_load(
    get_pycaret_data, metadata, modelstore, save_proc
):  # noqa # pylint: disable
    _, test_data = get_pycaret_data
    _model = save_proc(metadata)
    assert _model.info.metadata is not None
    assert_have_file_extension(_model.path, ".pkl")

    pycaret_loaded = bentoml.pycaret.load(
        _model.tag,
        model_store=modelstore,
    )
    assert isinstance(pycaret_loaded, sklearn.pipeline.Pipeline)
    assert predict_model(pycaret_loaded, data=test_data)["Score"][0] == 0.7609


@pytest.mark.parametrize("exc", [BentoMLException])
def test_pycaret_load_exc(wrong_module, exc, modelstore):
    with pytest.raises(exc):
        bentoml.pycaret.load(wrong_module, model_store=modelstore)


def test_pycaret_runner_setup_run_batch(get_pycaret_data, modelstore, save_proc):
    _, test_data = get_pycaret_data
    info = save_proc(None)

    runner = bentoml.pycaret.load_runner(tag=info.tag, model_store=modelstore)

    assert info.tag in runner.required_models
    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == 1

    assert runner.run_batch(test_data)["Score"][0] == 0.7609
    assert isinstance(runner._model, sklearn.pipeline.Pipeline)
