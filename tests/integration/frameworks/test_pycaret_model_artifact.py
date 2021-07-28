from pycaret.classification import create_model, predict_model
from pycaret.classification import setup as pycaret_setup
from pycaret.classification import tune_model
from pycaret.datasets import get_data

from bentoml.pycaret import PycaretModel
from tests._internal.helpers import assert_have_file_extension


def set_dataset():
    dataset = get_data("credit")
    train_data = dataset.sample(frac=0.95, random_state=786)
    pred_data = dataset.drop(train_data.index)
    train_data.reset_index(inplace=True, drop=True)
    pred_data.reset_index(inplace=True, drop=True)
    return train_data, pred_data


def predict_data(model, data):
    pred = predict_model(model, data=data)
    return pred.iloc[0]["Score"]


def test_pycaret_save_load(tmpdir):
    data, expected = set_dataset()
    _ = pycaret_setup(
        data=data, target="default", silent=True, html=False, session_id=420, n_jobs=1,
    )
    dt = create_model("dt")  # create a decision tree classifier
    tdt = tune_model(dt)

    PycaretModel(tdt).save(tmpdir)
    assert_have_file_extension(tmpdir, ".pkl")

    pycaret_loaded = PycaretModel.load(tmpdir)
    assert predict_data(pycaret_loaded, expected) == predict_data(tdt, expected)
