import os
import typing as t

import numpy as np
import pandas as pd
import psutil
import pytest
import statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults

import bentoml.statsmodels
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension

# fmt: off
test_df = pd.DataFrame([[0, 0, 1, 1]])
test_df2 = np.array([0, 0, 1, 1])

# fmt: on
if t.TYPE_CHECKING:
    from bentoml._internal.models import Model, ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


def predict_df(model: t.Any, df: pd.DataFrame):
    return model.predict(int(df.iat[0, 0]))


@pytest.fixture(scope="module")
def save_proc(
    modelstore: "ModelStore",
    holt_model,
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "Model"]:
    def _(metadata, holt_model) -> "Model":
        tag = bentoml.statsmodels.save(
            TEST_MODEL_NAME, holt_model, metadata=metadata, model_store=modelstore
        )
        model = modelstore.get(tag)
        return model

    return _


def wrong_module(modelstore: "ModelStore", holt_model):
    with modelstore.register(
        "wrong_module",
        module=__name__,
        options=None,
        framework_context=None,
        metadata=None,
    ) as ctx:
        holt_model.save(os.path.join(ctx.path, "saved_model.pkl"))
        return ctx.tag


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
def test_statsmodels_save_load(
    metadata, modelstore, holt_model
):  # noqa # pylint: disable
    tag = bentoml.statsmodels.save(
        TEST_MODEL_NAME, holt_model, metadata=metadata, model_store=modelstore
    )
    info = modelstore.get(tag)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".pkl")

    statsmodels_loaded = bentoml.statsmodels.load(tag, model_store=modelstore)

    assert isinstance(
        statsmodels_loaded,
        statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper,
    )

    np.testing.assert_array_equal(holt_model.predict(), statsmodels_loaded.predict())


@pytest.mark.parametrize("exc", [BentoMLException])
def test_get_model_info_exc(exc, modelstore, holt_model):
    tag = wrong_module(modelstore, holt_model)
    with pytest.raises(exc):
        bentoml.statsmodels._get_model_info(tag, model_store=modelstore)


def test_statsmodels_runner_setup_run_batch(modelstore, save_proc, holt_model):
    info = save_proc(None, holt_model)
    runner = bentoml.statsmodels.load_runner(
        info.tag, predict_fn_name="predict", model_store=modelstore
    )

    assert info.tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1

    res_pd = runner.run_batch(test_df)
    res_np = runner.run_batch(test_df2)

    expected_res = predict_df(holt_model, test_df)
    assert all(res_pd == expected_res)
    assert all(res_np == expected_res)


@pytest.mark.gpus
def test_statsmodels_runner_setup_on_gpu(modelstore, save_proc):
    info = save_proc(None)
    resource_quota = dict(gpus=0, cpu=0.4)
    runner = bentoml.statsmodels.load_runner(
        info.tag, model_store=modelstore, resource_quota=resource_quota
    )

    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == 1


# runner.run
# runner.run_batch
