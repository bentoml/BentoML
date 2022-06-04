import spacy
import pytest

import bentoml
from tests.integration.frameworks.spacy.test_spacy_impl import test_json

MODEL_NAME = __name__.split(".")[-1]


def test_spacy_runner_setup_run_batch(
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
    "backend",
    ["pytorch", "tensorflow"],
)
def test_spacy_runner_setup_run_batch_on_gpu(
    spacy_model: spacy.language.Language,
    backend: str,
):
    tag = bentoml.spacy.save(MODEL_NAME, spacy_model)
    runner = bentoml.spacy.load_runner(
        tag,
        backend_options=backend,
    )
    assert tag in runner.required_models
    assert runner.num_replica == 1

    res = runner.run_batch(test_json["text"])
    for i in res:
        assert i.text == test_json["text"]
