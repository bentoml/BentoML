from tests.bento_service_examples.lightgbm_service import LgbModelService
import numpy as np
import lightgbm as lgb
from pandas import DataFrame
import pytest
import bentoml
from bentoml.yatai.client import YataiClient


@pytest.fixture()
def lightgbm_model_service_class():
    LgbModelService._bento_service_bundle_path = None
    LgbModelService._bento_service_bundle_version = None
    return LgbModelService


def get_trained_lgbm_model():
    data = lgb.Dataset(np.array([[0]]), label=np.array([0]))
    model = lgb.train({}, data, 100)
    return model


def test_lgbm_artifact_pack():
    model = get_trained_lgbm_model()
    svc = LgbModelService()
    svc.pack("model", model)

    assert svc.predict(DataFrame([[0]])) == [0]

    saved_path = svc.save()

    loaded_svc = bentoml.load(saved_path)

    assert loaded_svc.predict(DataFrame([[0]])) == [0]

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(svc.name, svc.version)
