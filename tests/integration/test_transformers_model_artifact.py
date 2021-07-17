import json

import pytest
from transformers import AutoModelWithLMHead, AutoTokenizer

import bentoml
from tests import (
    TransformersGPT2TextGenerator,
    build_api_server_docker_image,
    run_api_server_docker_container,
)

test_sentence = {"text": "A Bento box is a "}

result = (
    "A Bento box is a urn that is used to store the contents of a Bento box. "
    + "It is usually used to store the contents of a Bento box in a storage container."
    + "\n\nThe Bento box is a small, rectangular"
)


@pytest.fixture(scope="module")
def transformers_svc():
    """Return a Transformers BentoService."""
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests
    TransformersGPT2TextGenerator._bento_service_bundle_path = None
    TransformersGPT2TextGenerator._bento_service_bundle_version = None

    svc = TransformersGPT2TextGenerator()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )
    model_artifact = {"model": model, "tokenizer": tokenizer}
    svc.pack("gptModel", model_artifact)
    return svc


@pytest.fixture(scope="module")
def transformers_svc_saved_dir(tmp_path_factory, transformers_svc):
    """Save a Transformers BentoService and return the saved directory."""
    # Must be called at least once before saving so that layers are built
    # See: https://github.com/tensorflow/tensorflow/issues/37439
    transformers_svc.predict(test_sentence)

    tmpdir = str(tmp_path_factory.mktemp("tf2_svc"))
    transformers_svc.save_to_dir(tmpdir)
    return tmpdir


@pytest.fixture()
def transformers_svc_loaded(transformers_svc_saved_dir):
    """Return a Transformers BentoService that has been saved and loaded."""
    return bentoml.load(transformers_svc_saved_dir)


@pytest.fixture()
def transformers_image(transformers_svc_saved_dir):
    with build_api_server_docker_image(
        transformers_svc_saved_dir, "tranformers_example_service"
    ) as image:
        yield image


@pytest.fixture()
def transformers_host(transformers_image):
    with run_api_server_docker_container(transformers_image, timeout=500) as host:
        yield host


def test_transformers_artifact(transformers_svc):
    assert (
        transformers_svc.predict(test_sentence) == result
    ), "Inference on unsaved Transformers artifact does not match expected"


def test_tensorflow_2_artifact_loaded(transformers_svc_loaded):
    assert (
        transformers_svc_loaded.predict(test_sentence) == result
    ), "Inference on saved and loaded Transformers artifact does not match expected"


@pytest.mark.asyncio
async def test_transformers_artifact_with_docker(transformers_host):
    await pytest.assert_request(
        "POST",
        f"http://{transformers_host}/predict",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps(test_sentence),
        assert_status=200,
    )
