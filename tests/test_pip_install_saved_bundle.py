import os
import sys
import json

import pandas as pd

from bentoml.configuration import get_bentoml_deploy_version


def test_pip_install_saved_bentoservice_bundle(bento_bundle_path, tmpdir):
    import subprocess

    install_path = str(tmpdir.mkdir("pip_local"))
    bentoml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    stdout = subprocess.check_output(
        ["pip", "install", "-U", "--target={}".format(install_path), bento_bundle_path]
    ).decode('utf-8')

    assert "Processing {}".format(bento_bundle_path) in stdout
    assert "Collecting bentoml=={}".format(get_bentoml_deploy_version()) in stdout
    assert "Successfully built TestBentoService" in stdout

    # ensure BentoML is installed as dependency
    assert os.path.isfile(os.path.join(install_path, "bin/bentoml"))
    assert os.path.isdir(os.path.join(install_path, "bentoml"))

    sys.path.insert(0, install_path)
    TestBentoService = __import__("TestBentoService")
    sys.path.remove(install_path)

    svc = TestBentoService.load()
    res = svc.predict_dataframe(pd.DataFrame(pd.DataFrame([1], columns=["col1"])))
    assert res == 1

    # pip install should place cli entry script under target/bin directory
    cli_bin_path = os.path.join(install_path, "bin", "TestBentoService")
    assert os.path.isfile(cli_bin_path)

    # add install_path and local bentoml module to PYTHONPATH to make them
    # available in subprocess call
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path + [install_path, bentoml_path])

    output = subprocess.check_output(
        [cli_bin_path, "--quiet", "info"], env=env
    ).decode()
    output = json.loads(output)
    assert output["name"] == "TestBentoService"
    assert output["version"] == svc.version
    assert "predict_dataframe" in map(lambda x: x["name"], output["apis"])

    output = subprocess.check_output(
        [cli_bin_path, "--quiet", "open-api-spec"], env=env
    ).decode()
    output = json.loads(output)
    assert output["info"]["version"] == svc.version
