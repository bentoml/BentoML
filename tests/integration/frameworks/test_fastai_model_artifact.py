import os

import numpy as np
from fastai.data.block import DataBlock
from fastai.learner import Learner

from bentoml.fastai import FastaiModel

from ..._internal.bento_services.fastai import Loss, Model, get_items, test_df


def pack_models(path: str) -> None:

    model = Model()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)

    FastaiModel(learner).save(path)


def predict_df(model: "Learner", df: "pd.DataFrame"):
    input_df = df.to_numpy().astype(np.float32)
    _, _, res = model.predict(input_df)
    return res.squeeze().item()


def test_fastai_save_pack(tmpdir):
    pack_models(tmpdir)
    assert os.path.exists(FastaiModel.get_path(tmpdir, ".pkl"))

    loaded_fastai: "Learner" = FastaiModel.load(tmpdir)
    assert predict_df(loaded_fastai, test_df) == 5.0
