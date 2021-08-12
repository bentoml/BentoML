import pytest
# from bentoml._internal.repository import LocalBundleManagement
import tempfile
import os
from pathlib import Path
import bentoml
LocalBundleManagement = bentoml._internal.repository


@pytest.fixture
def bundles_dir(tmpdir):
    test_bundles = [
        ("name1", "version1", None, ["model1", "model2", "model3"]),
        ("name1", "version2", None, ["model1", "model2", "model3"]),
        ("name2", "version1", None, ["model1", "model2", "model3"]),
        ("name2", "version2", None, ["model1", "model2", "model3"]),
    ]
    for test_bundle in test_bundles:
        name, version, _, models = test_bundle
        for model in models:
            tmpdir.mkdir(name).mkdir(version).mkdir(model)
    return tmpdir


# test list all
def test_list_all():
    LBM = LocalBundleManagement()
    '''
    #bundles_dir = tempfile.TemporaryDirectory()
    #bundles_dir_name = bundles_dir.name
    #LBM._bundle_store_path = bundles_dir_name

    # set up temporary bundles

    test_bundles = [
        ("name1", "version1", None, ["model1", "model2", "model3"]),
        ("name1", "version2", None, ["model1", "model2", "model3"]),
        ("name2", "version1", None, ["model1", "model2", "model3"]),
        ("name2", "version2", None, ["model1", "model2", "model3"]),
    ]
    for test_bundle in test_bundles:
        name, version, _, models = test_bundle
        for model in models:
            p = Path(os.path.join(bundles_dir_name, name, version, model))
            p.mkdir(parents=True, exist_ok=True)
    '''
    LBM._bundle_store_path = bundles_dir()

    bundles = LBM.list()
    bundles = [(n,v,None,m) for n,v,_,m in bundles]

    expected_bundles = [
        ("name1", "version1", None, ["model1", "model2", "model3"]),
        ("name1", "version2", None, ["model1", "model2", "model3"]),
        ("name2", "version1", None, ["model1", "model2", "model3"]),
        ("name2", "version2", None, ["model1", "model2", "model3"]),
    ]
    assert bundles == expected_bundles

    bundles_dir.cleanup()

# test list all with name
def test_list_all_name():
    LBM = LocalBundleManagement()
    bundles_dir = tempfile.TemporaryDirectory()
    bundles_dir_name = bundles_dir.name
    LBM._bundle_store_path = bundles_dir_name

    # set up temporary bundles
    test_bundles = [
        ("name1", "version1", None, ["model1", "model2", "model3"]),
        ("name1", "version2", None, ["model1", "model2", "model3"]),
        ("name2", "version1", None, ["model1", "model2", "model3"]),
        ("name2", "version2", None, ["model1", "model2", "model3"]),
    ]
    for test_bundle in test_bundles:
        name, version, _, models = test_bundle
        for model in models:
            p = Path(os.path.join(bundles_dir_name, name, version, model))
            p.mkdir(parents=True, exist_ok=True)

    bundles = LBM.list("name1")
    bundles = [(n,v,None,m) for n,v,_,m in bundles]

    expected_bundles = [
        ("name1", "version1", None, ["model1", "model2", "model3"]),
        ("name1", "version2", None, ["model1", "model2", "model3"]),
    ]
    assert bundles == expected_bundles

    bundles_dir.cleanup()

# test get tag
def test_get_tag():
    LBM = LocalBundleManagement()
    bundles_dir = tempfile.TemporaryDirectory()
    bundles_dir_name = bundles_dir.name
    LBM._bundle_store_path = bundles_dir_name

    # set up temporary bundles
    test_bundle = ("name1", "version1", None, ["model1"])
    p = Path(os.path.join(bundles_dir_name, "name1", "version1", "model1"))
    p.mkdir(parents=True, exist_ok=True)


    bundle = list(LBM.get("name1:version1"))
    bundle[2] = None # don't check time
    bundle = tuple(bundle)
    
    assert bundle == test_bundle

    bundles_dir.cleanup()

# test get tag doesn't exist
def test_get_tag_doesnt_exist():
    LBM = LocalBundleManagement()
    bundles_dir = tempfile.TemporaryDirectory()
    bundles_dir_name = bundles_dir.name
    LBM._bundle_store_path = bundles_dir_name

    bundle = LBM.get('THIS_NAME:DOES_NOT_EXIST')

    assert bundle == None

    bundles_dir.cleanup()

# delete with tag, name, all, and without --yes
def delete_without_yes():
    LBM = LocalBundleManagement()
    LBM._confirm = lambda s: True
    bundles_dir = tempfile.TemporaryDirectory()
    bundles_dir_name = bundles_dir.name
    LBM._bundle_store_path = bundles_dir_name

    # set up temporary bundles
    test_bundles = [
        ("name1", "version1", None, ["model1", "model2", "model3"]),
        ("name1", "version2", None, ["model1", "model2", "model3"]),
        ("name2", "version1", None, ["model1", "model2", "model3"]),
        ("name2", "version2", None, ["model1", "model2", "model3"]),
    ]
    for test_bundle in test_bundles:
        name, version, _, models = test_bundle
        for model in models:
            p = Path(os.path.join(bundles_dir_name, name, version, model))
            p.mkdir(parents=True, exist_ok=True)

    LBM.delete(tag="name1:version1")
    # TODO: test tag deleted
    assert not LBM.get("name1:version1")
    LBM.delete(name="name1")
    # TODO: test name deleted
    assert not LBM.list("name1")
    LBM.delete()
    # TODO: test all deleted
    assert not LBM.list()

    bundles_dir.cleanup()

# delete with tag, name, all, and with --yes
def delete_with_yes():
    LBM = LocalBundleManagement()
    bundles_dir = tempfile.TemporaryDirectory()
    bundles_dir_name = bundles_dir.name
    LBM._bundle_store_path = bundles_dir_name

    # set up temporary bundles
    test_bundles = [
        ("name1", "version1", None, ["model1", "model2", "model3"]),
        ("name1", "version2", None, ["model1", "model2", "model3"]),
        ("name2", "version1", None, ["model1", "model2", "model3"]),
        ("name2", "version2", None, ["model1", "model2", "model3"]),
    ]
    for test_bundle in test_bundles:
        name, version, _, models = test_bundle
        for model in models:
            p = Path(os.path.join(bundles_dir_name, name, version, model))
            p.mkdir(parents=True, exist_ok=True)

    LBM.delete(tag="name1:version1", skip_confirmation=True)
    # TODO: test tag deleted
    assert not LBM.get("name1:version1")
    LBM.delete(name="name1", skip_confirmation=True)
    # TODO: test name deleted
    assert not LBM.list("name1")
    LBM.delete(skip_confirmation=True)
    # TODO: test all deleted
    assert not LBM.list()

    bundles_dir.cleanup()