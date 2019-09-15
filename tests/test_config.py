import os
import contextlib

from configparser import ConfigParser

from bentoml.configuration.configparser import BentoMLConfigParser


@contextlib.contextmanager
def env_vars(**kwargs):
    original = {}
    for key, value in kwargs.items():
        original[key] = os.environ.get(key)
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    yield
    for key, value in original.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


def test_conf_access_hierachy():
    test_default_config = b"""\
[test]
a = 123
b = true
c = false
d = 1.01
e = value

[test2]
foo = bar
    """.decode(
        "utf-8"
    )
    config = BentoMLConfigParser(default_config=test_default_config)

    assert config["test"].getint("a") == 123
    assert config["test"].getboolean("b")
    assert not config["test"].getboolean("c")
    assert config["test"].getfloat("d") == 1.01
    assert config["test"].get("e") == "value"
    assert config["test2"].get("foo") == "bar"

    config.read_string(
        b"""\
[test]
c = true
e = value ii
    """.decode(
            "utf-8"
        )
    )

    assert config["test"].getboolean("c")
    assert config["test"].get("e") == "value ii"

    with env_vars(
        BENTOML__TEST__C="false",
        BENTOML__TEST__E="value iii",
        BENTOML__TEST2__FOO="new bar",
    ):
        assert not config["test"].getboolean("c")
        assert config["test"].get("e") == "value iii"
        assert config["test2"].get("foo") == "new bar"

        config_copy = ConfigParser()
        config_copy.read_dict(config.as_dict())

        assert config_copy["test"].getint("a") == 123
        assert config_copy["test"].getboolean("b")
        assert not config_copy["test"].getboolean("c")
        assert config_copy["test"].getfloat("d") == 1.01
        assert config_copy["test"].get("e") == "value iii"
        assert config_copy["test2"].get("foo") == "new bar"
