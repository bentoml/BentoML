import os
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pyspark.files import SparkFiles
from pyspark.sql.session import SparkSession

import bentoml
from src.bentoml._internal.spark import _load_bento
from src.bentoml._internal.spark import _distribute_bento
from bentoml._internal.bento.bento import Bento
from bentoml._internal.bento.bento import BentoStore
from src.bentoml._internal.service.loader import load_bento
from src.bentoml._internal.service.loader import _load_bento as load_bento_service
from src.bentoml._internal.service.loader import on_load_bento
from src.bentoml._internal.service.loader import import_service
from bentoml._internal.configuration.containers import BentoMLContainer


def test_distribute_bento(test_bento: Bento) -> None:

    # TODO: pair with Seoyeon
    # breakpoint()
    test_bento.info.labels["spark"] = "test"
    test_bento.flush_info()

    SparkSession = Mock()
    SparkSession.sparkContext.addFile.return_value = None

    _distribute_bento(SparkSession, test_bento)

    SparkSession.sparkContext.addFile.assert_called_once()
    export_path = SparkSession.sparkContext.addFile.call_args.args[0]
    exported_bento = bentoml.bentos.import_bento(export_path)

    assert exported_bento.info.name == test_bento.info.name
    assert exported_bento.info.tag == test_bento.info.tag
    assert exported_bento.info.labels == test_bento.info.labels


def test_load_bento(test_bento: Bento, bento_store: BentoStore) -> None:

    SparkFiles.get = Mock()
    SparkFiles.get.return_value = os.path.abspath(
        "tests/unit/_internal/spark/test_bento.bento"
    )  # .get returns abspath

    breakpoint()
    with BentoMLContainer.bento_store.patch(bento_store):
        _load_bento(test_bento.info.tag)

    SparkFiles.get.assert_called_once()
    bento_file = SparkFiles.get.call_args.args[0]

    assert (
        bento_file == f"{test_bento.info.tag.name}-{test_bento.info.tag.version}.bento"
    )
