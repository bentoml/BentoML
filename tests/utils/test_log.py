import logging
import os
import tempfile

from bentoml.utils.log import configure_logging


def test_configure_logging_default():
    configure_logging()

    bentoml_logger = logging.getLogger("bentoml")
    assert bentoml_logger.level == logging.INFO
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 2
    assert bentoml_logger.handlers[0].name == "console"
    assert bentoml_logger.handlers[1].name == "local"

    prediction_logger = logging.getLogger("bentoml.prediction")
    assert prediction_logger.level == logging.INFO
    assert prediction_logger.propagate is False
    assert len(prediction_logger.handlers) == 1
    assert prediction_logger.handlers[0].name == "prediction"

    feedback_logger = logging.getLogger("bentoml.feedback")
    assert feedback_logger.level == logging.INFO
    assert feedback_logger.propagate is False
    assert len(feedback_logger.handlers) == 1
    assert feedback_logger.handlers[0].name == "feedback"


def test_configure_logging_custom_level():
    configure_logging(logging.ERROR)

    bentoml_logger = logging.getLogger("bentoml")
    assert bentoml_logger.level == logging.ERROR
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 2
    assert bentoml_logger.handlers[0].name == "console"
    assert bentoml_logger.handlers[1].name == "local"

    prediction_logger = logging.getLogger("bentoml.prediction")
    assert prediction_logger.level == logging.INFO
    assert prediction_logger.propagate is False
    assert len(prediction_logger.handlers) == 1
    assert prediction_logger.handlers[0].name == "prediction"

    feedback_logger = logging.getLogger("bentoml.feedback")
    assert feedback_logger.level == logging.INFO
    assert feedback_logger.propagate is False
    assert len(feedback_logger.handlers) == 1
    assert feedback_logger.handlers[0].name == "feedback"


def test_configure_logging_file_disabled():
    os.environ["BENTOML__LOGGING__CONSOLE_LOGGING_ENABLED"] = "false"

    configure_logging()

    bentoml_logger = logging.getLogger("bentoml")
    assert bentoml_logger.level == logging.INFO
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 1
    assert bentoml_logger.handlers[0].name == "local"

    prediction_logger = logging.getLogger("bentoml.prediction")
    assert prediction_logger.level == logging.INFO
    assert prediction_logger.propagate is False
    assert len(prediction_logger.handlers) == 1
    assert prediction_logger.handlers[0].name == "prediction"

    feedback_logger = logging.getLogger("bentoml.feedback")
    assert feedback_logger.level == logging.INFO
    assert feedback_logger.propagate is False
    assert len(feedback_logger.handlers) == 1
    assert feedback_logger.handlers[0].name == "feedback"

    del os.environ["BENTOML__LOGGING__CONSOLE_LOGGING_ENABLED"]


def test_configure_logging_console_disabled():
    os.environ["BENTOML__LOGGING__FILE_LOGGING_ENABLED"] = "false"

    configure_logging()

    bentoml_logger = logging.getLogger("bentoml")
    assert bentoml_logger.level == logging.INFO
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 1
    assert bentoml_logger.handlers[0].name == "console"

    prediction_logger = logging.getLogger("bentoml.prediction")
    assert prediction_logger.level == logging.INFO
    assert prediction_logger.propagate is False
    assert len(prediction_logger.handlers) == 0

    feedback_logger = logging.getLogger("bentoml.feedback")
    assert feedback_logger.level == logging.INFO
    assert feedback_logger.propagate is False
    assert len(feedback_logger.handlers) == 0

    del os.environ["BENTOML__LOGGING__FILE_LOGGING_ENABLED"]


def test_configure_logging_yaml():
    logging_config = tempfile.NamedTemporaryFile(delete=False)
    logging_config.write(
        b"""
version: 1
disable_existing_loggers: False
formatters:
    test_formatter:
        format: '[%(asctime)s] %(levelname)s - %(message)s'
handlers:
    test_handler:
        level: WARN
        formatter: test_formatter
        class: logging.StreamHandler
        stream: ext://sys.stdout
loggers:
    test_logger:
        handlers: [test_handler]
        level: WARN
        propagate: False"""
    )
    logging_config.close()

    os.environ["BENTOML__LOGGING__LOGGING_CONFIG"] = logging_config.name

    configure_logging()

    bentoml_logger = logging.getLogger("test_logger")
    assert bentoml_logger.level == logging.WARN
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 1
    assert bentoml_logger.handlers[0].name == "test_handler"

    os.remove(logging_config.name)
    del os.environ["BENTOML__LOGGING__LOGGING_CONFIG"]
