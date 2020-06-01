import bentoml
from bentoml.artifact import JSONArtifact


@bentoml.artifacts([JSONArtifact("hparams")])
class ExampleServiceWithJSONArtifact(bentoml.BentoService):
    pass


def test_json_artifact_simple_service_round_trip(tmp_path):
    service = ExampleServiceWithJSONArtifact()
    # 'hparams' = hyperparameters used at training time.
    service.pack("hparams", {
        "lr": 1e-3,
        "patience": 10,
        "decay": 0.1,
        "batch_size": 16
    })
    service.save_to_dir(str(tmp_path))
    del service
    new_service = bentoml.bundler.load(str(tmp_path))
    assert isinstance(new_service.artifacts.hparams, (dict, list, int, float, str))
