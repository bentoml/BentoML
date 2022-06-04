import typing as t

import numpy as np
import pandas as pd
import psutil
import pytest

import bentoml
import statsmodels
import bentoml.models
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension
from statsmodels.tsa.holtwinters import HoltWintersResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# fmt: off
test_df = pd.DataFrame([[0, 0, 1, 1]])
test_df2 = np.array([0, 0, 1, 1])

# fmt: on
if t.TYPE_CHECKING:
    from bentoml._internal.models import Model

TEST_MODEL_NAME = __name__.split(".")[-1]


def predict_df(model: t.Any, df: pd.DataFrame):
    return model.predict(int(df.iat[0, 0]))


@pytest.fixture(scope="module")
def save_proc(
    holt_model,
) -> t.Callable[
    [
        t.Dict[str, t.Any],
        "HoltWintersResults",
        t.Optional[t.Dict[str, str]],
        t.Optional[t.Dict[str, t.Any]],
    ],
    "Model",
]:
    def _(metadata, holt_model, labels=None, custom_objects=None) -> "Model":
        tag = bentoml.statsmodels.save(
            TEST_MODEL_NAME,
            holt_model,
            metadata=metadata,
            labels=labels,
            custom_objects=custom_objects,
        )
        model = bentoml.models.get(tag)
        return model

    return _


def wrong_module(holt_model):
    with bentoml.models.create(
        "wrong_module",
        module=__name__,
        options=None,
        context=None,
        metadata=None,
    ) as _model:
        holt_model.save(_model.path_of("saved_model.pkl"))
        return _model.tag


# exported from
#  https://colab.research.google.com/github/bentoml/gallery/blob/master/statsmodels_holt/bentoml_statsmodels.ipynb
@pytest.fixture(scope="session")
def holt_model() -> "HoltWintersResults":
    df: pd.DataFrame = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
    )

    # Taking a test-train split of 80 %
    train = df[0 : int(len(df) * 0.8)]
    test = df[int(len(df) * 0.8) :]

    # Pre-processing the  Month  field
    train.Timestamp = pd.to_datetime(train.Month, format="%m-%d")
    train.index = train.Timestamp
    test.Timestamp = pd.to_datetime(test.Month, format="%m-%d")
    test.index = test.Timestamp

    # fitting the model based on  optimal parameters
    return ExponentialSmoothing(
        np.asarray(train["Sales"]),
        seasonal_periods=7,
        trend="add",
        seasonal="add",
    ).fit()


@pytest.mark.parametrize(
    "metadata",
    [
        ({"model": "Statsmodels", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_statsmodels_save_load(metadata, holt_model):  # noqa # pylint: disable

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    tag = bentoml.statsmodels.save(
        TEST_MODEL_NAME,
        holt_model,
        metadata=metadata,
        labels=labels,
        custom_objects={"func": custom_f},
    )
    bentomodel = bentoml.models.get(tag)
    assert bentomodel.info.metadata is not None
    assert_have_file_extension(bentomodel.path, ".pkl")
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    statsmodels_loaded = bentoml.statsmodels.load(tag)

    assert isinstance(
        statsmodels_loaded,
        statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper,
    )

    np.testing.assert_array_equal(holt_model.predict(), statsmodels_loaded.predict())


@pytest.mark.parametrize("exc", [BentoMLException])
def test_load_model_exc(exc, holt_model):
    tag = wrong_module(holt_model)
    with pytest.raises(exc):
        bentoml._internal.frameworks.statsmodels.load(tag)


def test_statsmodels_runner_setup_run_batch(save_proc, holt_model):
    info = save_proc(None, holt_model)
    runner = bentoml.statsmodels.load_runner(info.tag, predict_fn_name="predict")

    assert info.tag in runner.required_models
    assert runner.num_replica == 1

    res_pd = runner.run_batch(test_df)
    res_np = runner.run_batch(test_df2)

    expected_res = predict_df(holt_model, test_df)
    assert all(res_pd == expected_res)
    assert all(res_np == expected_res)


@pytest.mark.gpus
def test_statsmodels_runner_setup_on_gpu(save_proc):
    bentomodel = save_proc(None)
    runner = bentoml.statsmodels.load_runner(bentomodel.tag)

    assert runner.num_replica == 1


# runner.run
# runner.run_batch
