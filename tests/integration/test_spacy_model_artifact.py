import pytest
import random
import spacy
import bentoml
from tests.bento_service_examples.spacy_classifier import SpacyModelService
from bentoml.yatai.client import YataiClient


@pytest.fixture()
def spacy_model_service_class():
    SpacyModelService._bento_service_bundle_path = None
    SpacyModelService._bento_service_bundle_version = None
    return SpacyModelService


TRAIN_DATA = [
    ("Google has changed the logo of its apps", {"entities": [(0, 6, "ORG")]}),
    ("Facebook has introduced a new app!", {"entities": [(0, 8, "ORG")]}),
    ("Amazon has partnered with small businesses.", {"entities": [(0, 6, "ORG")]}),
]


def get_trained_model():
    model = spacy.blank("en")

    if 'ner' not in model.pipe_names:
        ner = model.create_pipe('ner')
        model.add_pipe(ner, last=True)
    else:
        ner = model.get_pipe('ner')

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in model.pipe_names if pipe != 'ner']

    with model.disable_pipes(*other_pipes):
        optimizer = model.begin_training()
        for iteration in range(10):
            random.shuffle(TRAIN_DATA)
            for text, annotation in TRAIN_DATA:
                model.update([text], [annotation], sgd=optimizer)

    return model


def test_spacy_artifact_pack(spacy_model_service_class):
    nlp = get_trained_model()
    svc = spacy_model_service_class()
    svc.pack('nlp', nlp)

    test_json = {"text": "Google cloud is now a separate entity."}
    assert [(ent.text, ent.label_) for ent in svc.predict(test_json).ents] == [
        ('Google', 'ORG')
    ], 'Run inference before saving the artifact'

    saved_path = svc.save()

    loaded_svc = bentoml.load(saved_path)

    assert [(ent.text, ent.label_) for ent in loaded_svc.predict(test_json).ents] == [
        ('Google', 'ORG')
    ], 'Run inference before saving the artifact'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f'{svc.name}:{svc.version}')
