#!/usr/bin/env python3

from fastai.basics import URLs
from fastai.metrics import accuracy
from fastai.text.data import DataBlock
from fastai.text.data import TextBlock
from fastai.text.data import untar_data
from fastai.text.data import CategoryBlock
from fastai.text.models import AWD_LSTM
from fastai.text.learner import text_classifier_learner
from fastai.data.transforms import parent_label
from fastai.data.transforms import get_text_files
from fastai.data.transforms import GrandparentSplitter


def loading_learner():
    path = untar_data(URLs.IMDB)

    imdb = DataBlock(
        blocks=(TextBlock.from_folder(path), CategoryBlock),
        get_items=get_text_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(valid_name="test"),
    )
    dls = imdb.dataloaders(path)

    learner = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    return learner


learner = loading_learner()

if __name__ == "__main__":
    learner.fine_tune(4, 1e-2)

    learner.show_results()
