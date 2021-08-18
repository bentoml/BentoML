import multiprocessing
import os
import sys
from bentoml.configuration import (
    get_local_config_file,
    get_debug_mode,
    set_debug_mode,
)


def test_get_local_config_file():
    config_file = get_local_config_file()

    assert config_file is None

    os.environ["QWAK_BENTOML_CONFIG"] = "/tmp/bentoml.cfg"
    config_file = get_local_config_file()

    assert config_file == "/tmp/bentoml.cfg"

    del os.environ["QWAK_BENTOML_CONFIG"]


def test_get_debug_mode():
    os.environ["BENTOML_DEBUG"] = "TRUE"
    assert get_debug_mode()

    os.environ["BENTOML_DEBUG"] = "true"
    assert get_debug_mode()

    os.environ["BENTOML_DEBUG"] = "True"
    assert get_debug_mode()

    os.environ["BENTOML_DEBUG"] = "FALSE"
    assert not get_debug_mode()

    os.environ["BENTOML_DEBUG"] = "false"
    assert not get_debug_mode()

    os.environ["BENTOML_DEBUG"] = "False"
    assert not get_debug_mode()

    del os.environ["BENTOML_DEBUG"]
    assert not get_debug_mode()


def test_set_debug_mode():
    set_debug_mode(True)
    assert get_debug_mode()

    set_debug_mode(False)
    assert not get_debug_mode()

    del os.environ["BENTOML_DEBUG"]


def assert_debug_mode(enabled: bool):
    if get_debug_mode() is enabled:
        sys.exit(0)
    else:
        sys.exit(1)


def test_multiprocess_debug_mode():
    """get_debug_mode() and set_debug_mode() should preserve between processes"""

    set_debug_mode(True)
    assert get_debug_mode()

    process = multiprocessing.Process(
        target=assert_debug_mode, args=[True], daemon=True
    )
    process.start()
    process.join()

    assert process.exitcode == 0

    set_debug_mode(False)
    assert not get_debug_mode()

    process = multiprocessing.Process(
        target=assert_debug_mode, args=[False], daemon=True
    )
    process.start()
    process.join()

    assert process.exitcode == 0

    del os.environ["BENTOML_DEBUG"]
