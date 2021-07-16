import pytest
import tempfile
import contextlib
import bentoml
from tests.bento_services.fasttext_classifier import FasttextClassifier
from bentoml.yatai.client import YataiClient

import fasttext


@pytest.fixture()
def fasttext_classifier_class():
    FasttextClassifier._bento_service_bundle_path = None
    FasttextClassifier._bento_service_bundle_version = None
    return FasttextClassifier


test_json = {"text": "foo"}


def test_fasttext_artifact_pack(fasttext_classifier_class):
    @contextlib.contextmanager
    def _temp_filename_with_contents(contents):
        temporary_file = tempfile.NamedTemporaryFile(suffix=".txt", mode="w+")
        temporary_file.write(contents)
        # Set file pointer to beginning to ensure correct read
        temporary_file.seek(0)
        yield temporary_file.name
        temporary_file.close()

    with _temp_filename_with_contents("__label__bar foo") as filename:
        model = fasttext.train_supervised(input=filename)

    svc = fasttext_classifier_class()
    svc.pack('model', model)

    assert svc.predict(test_json)[0] == (
        '__label__bar',
    ), 'Run inference before saving the artifact'

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)
    assert loaded_svc.predict(test_json)[0] == (
        '__label__bar',
    ), 'Run inference after saving the artifact'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f'{svc.name}:{svc.version}')
