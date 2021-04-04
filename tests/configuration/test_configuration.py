import os
from bentoml.configuration import get_local_config_file, BENTOML_HOME


def test_get_local_config_file():
    config_file = get_local_config_file()

    assert config_file == os.path.join(BENTOML_HOME, "bentoml.cfg")

    os.environ["BENTOML_CONFIG"] = "/tmp/bentoml.cfg"
    config_file = get_local_config_file()

    assert config_file == "/tmp/bentoml.cfg"

    del os.environ["BENTOML_CONFIG"]
