import psutil
import pytest

import bentoml.spacy
from tests.integration.frameworks.spacy.test_spacy_impl import \
    spacy_model  # noqa
from tests.integration.frameworks.spacy.test_spacy_impl import test_json

MODEL_NAME = __name__.split(".")[-1]


def test_spacy_runner_setup_run_batch(modelstore, spacy_model):  # noqa: F811
    tag = bentoml.spacy.save(MODEL_NAME, spacy_model, model_store=modelstore)
    runner = bentoml.spacy.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.num_replica == 1
    assert runner.num_concurrency_per_replica == psutil.cpu_count()

    res = runner.run_batch([test_json["text"]])
    for i in res:
        assert i.text == test_json["text"]


@pytest.mark.gpus
@pytest.mark.parametrize("dev, backend", [(0, "pytorch"), (0, "tensorflow")])
def test_spacy_runner_setup_on_gpu(spacy_model, modelstore, dev, backend):  # noqa: F811
    tag = bentoml.spacy.save(MODEL_NAME, spacy_model, model_store=modelstore)
    runner = bentoml.spacy.load_runner(
        tag, model_store=modelstore, gpu_device_id=dev, backend_options=backend
    )

    assert tag in runner.required_models
    assert runner.num_replica == 1
    assert runner.num_concurrency_per_replica == 1

    res = runner.run_batch([test_json["text"]])
    for i in res:
        assert i.text == test_json["text"]
