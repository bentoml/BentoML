import json
import os
import sys

import pandas as pd
import psutil
import pytest


def format_path(path_str):
    if psutil.WINDOWS:
        return path_str.lower().replace('/', '\\')
    return path_str


@pytest.mark.skipif('not psutil.POSIX')
def test_pip_install_saved_bentoservice_bundle(bento_bundle_path, tmpdir):
    import subprocess
    from pip._internal.cli.main import main as pipmain

    install_path = str(tmpdir.mkdir("pip_local"))
    bentoml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    assert (
        pipmain(
            ["install", "-U", "--target={}".format(install_path), bento_bundle_path]
        )
        == 0
    ), 'saved bundle successfully installed'

    # ensure BentoML is installed as dependency
    if psutil.WINDOWS:
        assert os.path.isfile(os.path.join(install_path, "bin", "bentoml.exe"))
    else:
        assert os.path.isfile(os.path.join(install_path, "bin", "bentoml"))

    assert os.path.isdir(os.path.join(install_path, "bentoml"))

    sys.path.insert(0, install_path)
    ExampleBentoService = __import__("ExampleBentoService")
    sys.path.remove(install_path)

    svc = ExampleBentoService.load()
    res = svc.predict_dataframe(pd.DataFrame(pd.DataFrame([1], columns=["col1"])))
    assert (res == 2).all()

    res = svc.predict_dataframe_v1(pd.DataFrame(pd.DataFrame([1], columns=["col1"])))
    assert (res == 2).all()

    # pip install should place cli entry script under target/bin directory
    if psutil.WINDOWS:
        cli_bin_path = os.path.join(install_path, "bin", "ExampleBentoService.exe")
    else:
        cli_bin_path = os.path.join(install_path, "bin", "ExampleBentoService")
    assert os.path.isfile(cli_bin_path)

    # add install_path and local bentoml module to PYTHONPATH to make them
    # available in subprocess call
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path + [install_path, bentoml_path])

    output = subprocess.check_output(
        [cli_bin_path, "info", "--quiet"], env=env
    ).decode()
    output = json.loads(output)
    assert output["name"] == "ExampleBentoService"
    assert output["version"] == svc.version
    assert "predict_dataframe" in map(lambda x: x["name"], output["apis"])

    output = subprocess.check_output(
        [cli_bin_path, "open-api-spec", "--quiet"], env=env
    ).decode()
    output = json.loads(output)
    assert output["info"]["version"] == svc.version
