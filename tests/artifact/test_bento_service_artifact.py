import pytest
from bentoml.artifact import BentoServiceArtifact, SklearnModelArtifact
from bentoml.exceptions import FailedPrecondition


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


def test_artifact_states(tmp_path):
    from sklearn import svm
    from sklearn import datasets

    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    artifact_name = 'test_model'
    a1 = SklearnModelArtifact(artifact_name)

    # verify initial states
    assert not a1.loaded
    assert not a1.packed
    assert not a1.is_ready

    # verify that get will fail
    with pytest.raises(FailedPrecondition):
        a1.get()
    # verify that save will fail
    with pytest.raises(FailedPrecondition):
        a1.save('anywhere')

    # verify states after pack
    a1.pack(clf)
    assert not a1.loaded
    assert a1.packed
    assert a1.is_ready
    assert isinstance(a1.get(), svm.SVC)

    # Test save and load artifact
    # 1. save the packed artifact to tempdir
    a1.save(tmp_path)
    # 2. create a new artifact with same name
    a2 = SklearnModelArtifact(artifact_name)
    # 3. load a2 from tempdir
    a2.load(tmp_path)

    # verify states after load
    assert a2.loaded
    assert a2.is_ready
    assert a2.packed  # this dependes on the artifact implementation, in the case of
    # Sklearn artifact, the `load` method invokes `pack` internally
    assert isinstance(a2.get(), svm.SVC)
    assert all(a1.get().predict(X) == a2.get().predict(X))
