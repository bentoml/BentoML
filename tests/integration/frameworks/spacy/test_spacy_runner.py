import sys
from typing import Dict
from typing import TYPE_CHECKING

import spacy
import psutil
import pytest

import bentoml
from tests.integration.frameworks.spacy.test_spacy_impl import test_json

if TYPE_CHECKING:
    from bentoml._internal.models import ModelStore
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

MODEL_NAME = __name__.split(".")[-1]


def test_spacy_runner_setup_run_batch(
    modelstore: "ModelStore",
    spacy_model: spacy.language.Language,
):
    tag = bentoml.spacy.save(MODEL_NAME, spacy_model)
    runner = bentoml.spacy.load_runner(tag)

    assert tag in runner.required_models
    assert runner.num_replica == 1

    res = runner.run_batch([test_json["text"]])
    for i in res:
        assert i.text == test_json["text"]


@pytest.mark.gpus
@pytest.mark.parametrize(
    "dev, backend, resource_quota",
    [(0, "pytorch", {}), (0, "tensorflow", {"mem": "512Mi"})],
)
def test_spacy_runner_setup_run_batch_on_gpu(
    spacy_model: spacy.language.Language,
    modelstore: "ModelStore",
    dev: int,
    backend: Literal["pytorch", "tensorflow"],
    resource_quota: Dict[str, str],
):
    tag = bentoml.spacy.save(MODEL_NAME, spacy_model)
    runner = bentoml.spacy.load_runner(
        tag,
        backend_options=backend,
        resource_quota=resource_quota,
    )
    assert tag in runner.required_models
    assert runner.resource_quota.gpus == [str(dev)]
    assert runner.num_replica == 1

    res = runner.run_batch(test_json["text"])
    for i in res:
        assert i.text == test_json["text"]
