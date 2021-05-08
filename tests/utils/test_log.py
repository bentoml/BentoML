import logging

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
    assert len(prediction_logger.handlers) == 2
    assert prediction_logger.handlers[0].name == "console"
    assert prediction_logger.handlers[1].name == "prediction"

    feedback_logger = logging.getLogger("bentoml.feedback")
    assert feedback_logger.level == logging.INFO
    assert feedback_logger.propagate is False
    assert len(feedback_logger.handlers) == 2
    assert feedback_logger.handlers[0].name == "console"
    assert feedback_logger.handlers[1].name == "feedback"


def test_configure_logging_custom_level():
    configure_logging(logging_level=logging.ERROR)

    bentoml_logger = logging.getLogger("bentoml")
    assert bentoml_logger.level == logging.ERROR
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 2
    assert bentoml_logger.handlers[0].name == "console"
    assert bentoml_logger.handlers[1].name == "local"

    prediction_logger = logging.getLogger("bentoml.prediction")
    assert prediction_logger.level == logging.INFO
    assert prediction_logger.propagate is False
    assert len(prediction_logger.handlers) == 2
    assert prediction_logger.handlers[0].name == "console"
    assert prediction_logger.handlers[1].name == "prediction"

    feedback_logger = logging.getLogger("bentoml.feedback")
    assert feedback_logger.level == logging.INFO
    assert feedback_logger.propagate is False
    assert len(feedback_logger.handlers) == 2
    assert feedback_logger.handlers[0].name == "console"
    assert feedback_logger.handlers[1].name == "feedback"


def test_configure_logging_console_disabled():
    configure_logging(console_logging_enabled=False)

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


def test_configure_logging_file_disabled():
    configure_logging(file_logging_enabled=False)

    bentoml_logger = logging.getLogger("bentoml")
    assert bentoml_logger.level == logging.INFO
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 1
    assert bentoml_logger.handlers[0].name == "console"

    prediction_logger = logging.getLogger("bentoml.prediction")
    assert prediction_logger.level == logging.INFO
    assert prediction_logger.propagate is False
    assert len(prediction_logger.handlers) == 1
    assert prediction_logger.handlers[0].name == "console"

    feedback_logger = logging.getLogger("bentoml.feedback")
    assert feedback_logger.level == logging.INFO
    assert feedback_logger.propagate is False
    assert len(feedback_logger.handlers) == 1
    assert feedback_logger.handlers[0].name == "console"


def test_configure_logging_advanced():
    advanced_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "test_formatter": {"format": "[%(asctime)s] %(levelname)s - %(message)s"}
        },
        "handlers": {
            "test_handler": {
                "level": "WARN",
                "formatter": "test_formatter",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "test_logger": {
                "handlers": ["test_handler"],
                "level": "WARN",
                "propagate": False,
            }
        },
    }

    configure_logging(advanced_enabled=True, advanced_config=advanced_config)

    bentoml_logger = logging.getLogger("test_logger")
    assert bentoml_logger.level == logging.WARN
    assert bentoml_logger.propagate is False
    assert len(bentoml_logger.handlers) == 1
    assert bentoml_logger.handlers[0].name == "test_handler"
