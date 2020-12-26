Logging
=======

BentoML uses standard `Python logging <https://docs.python.org/3/howto/logging.html>`_ libraries and provides basic logging customization through `bentoml.cfg` under the BentoML home directory. See below for configurable fields and default values.

.. code-block:: init

    [logging]
    logging_config = {BENTOML_HOME}/logging.yaml

    level = INFO
    log_format = [%%(asctime)s] %%(levelname)s - %%(message)s
    dev_log_format = [%%(asctime)s] {{%%(filename)s:%%(lineno)d}} %%(levelname)s - %%(message)s

    # the base file directory where bentoml store all its log files
    base_log_dir = {BENTOML_HOME}/logs/

    log_request_image_files = True

    prediction_log_filename = prediction.log
    prediction_log_json_format = "%%(service_name)s %%(service_version)s %%(api)s %%(request_id)s %%(task)s %%(result)s %%(asctime)s"

    feedback_log_filename = feedback.log
    feedback_log_json_format = "%%(service_name)s %%(service_version)s %%(request_id)s %%(asctime)s"

    yatai_web_server_log_filename = yatai_web_server.log    

For advanced logging customization, user can provide full logging configuration in `logging.yaml`, placed under BentoML home directory. Please see below for an example of logging configuration in yaml format.

.. code-block:: yaml

    version: 1
    disable_existing_loggers: False
    formatters:
        console:
            format: '[%(asctime)s] %(levelname)s - %(message)s'
        dev: 
            format: '[%(asctime)s] {{%(filename)s:%(lineno)d}} %(levelname)s - %(message)s'
        prediction:
            (): pythonjsonlogger.jsonlogger.JsonFormatter
            fmt: '%(service_name)s %(service_version)s %(api)s %(request_id)s %(task)s %(result)s %(asctime)s'
        feedback:
            (): pythonjsonlogger.jsonlogger.JsonFormatter
            fmt: '%(service_name)s %(service_version)s %(request_id)s %(asctime)s'
    handlers:
        console:
            level: INFO
            formatter: console
            class: logging.StreamHandler
            stream: ext://sys.stdout
        local:
            level: INFO
            formatter: dev
            class: logging.handlers.RotatingFileHandler
            filename: '{BENTOML_HOME}/logs/active.log'
            maxBytes: 104857600
            backupCount: 2
        prediction:
            class: logging.handlers.RotatingFileHandler
            formatter: prediction
            level: INFO
            filename: '{BENTOML_HOME}/logs/prediction.log'
            maxBytes: 104857600
            backupCount: 10
        feedback:
            class: logging.handlers.RotatingFileHandler
            formatter: feedback
            level: INFO
            filename: '{BENTOML_HOME}/logs/feedback.log'
            maxBytes: 104857600
            backupCount: 10
    loggers:
        bentoml:
            handlers: [console, local]
            level: INFO
            propagate: False
        bentoml.prediction:
            handlers: [prediction, console]
            level: INFO
            propagate: False
        bentoml.feedback:
            handlers: [feedback, console]
            level: INFO
            propagate: False
