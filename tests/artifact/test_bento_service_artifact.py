import pytest
from bentoml.artifact import BentoServiceArtifact


def test_valid_artifact_name():
    name = "_test"
    artifact = BentoServiceArtifact(name)
    assert artifact.name == "_test"

    name = "test"
    artifact = BentoServiceArtifact(name)
    assert artifact.name == "test"

    name = "TEST00"
    artifact = BentoServiceArtifact(name)
    assert artifact.name == "TEST00"

    name = "_"
    artifact = BentoServiceArtifact(name)
    assert artifact.name == "_"

    name = "__"
    artifact = BentoServiceArtifact(name)
    assert artifact.name == "__"

    name = "thïsisàtèst"
    artifact = BentoServiceArtifact(name)
    assert artifact.name == "thïsisàtèst"

    name = "thisIs012Along__erTest"
    artifact = BentoServiceArtifact(name)
    assert artifact.name == "thisIs012Along__erTest"


def test_unvalid_artifact_name():
    name = ""
    with pytest.raises(ValueError) as e:
        BentoServiceArtifact(name)
    assert "Artifact name must be a valid python identifier" in str(e.value)

    name = "156test"
    with pytest.raises(ValueError) as e:
        BentoServiceArtifact(name)
    assert "Artifact name must be a valid python identifier" in str(e.value)

    name = "thisIs012Alo(*ng__erTest"
    with pytest.raises(ValueError) as e:
        BentoServiceArtifact(name)
    assert "Artifact name must be a valid python identifier" in str(e.value)

    name = "this is a test"
    with pytest.raises(ValueError) as e:
        BentoServiceArtifact(name)
    assert "Artifact name must be a valid python identifier" in str(e.value)
