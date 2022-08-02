from tempfile import NamedTemporaryFile

from bentoml._internal.configuration.containers import BentoMLConfiguration

OVERRIDE_RUNNERS = """
runners:
    batching:
        enabled: False
    logging:
        access:
            enabled: False
    test_runner_1:
        resources: system
    test_runner_2:
        resources: 
            cpu: 2 
    test_runner_batching:
        batching:
            enabled: True 
        logging:
            access:
                enabled: True
"""


def test_bentoml_configuration_runner_override():
    with NamedTemporaryFile(mode="w+", delete=True) as tmpfile:
        tmpfile.write(OVERRIDE_RUNNERS)
        bentoml_cfg = BentoMLConfiguration(override_config_file=tmpfile.name).as_dict()
        runner_cfg = bentoml_cfg["runners"]

        # test_runner_1
        test_runner_1 = runner_cfg["test_runner_1"]
        assert test_runner_1["batching"]["enabled"] is False
        assert test_runner_1["logging"]["access"]["enabled"] is False
