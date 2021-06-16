Logging
=======

BentoML produces `active.log`, `prediction.log`, and `feedback.log`. They are logged to both 
the console and file system under :code:`$BENTOML_HOME/logs` directory by default.

+----------------+-------------------------------------------------+
| active.log     | Logs generated from BentoML CLI and Python APIs |
+----------------+-------------------------------------------------+
| prediction.log | Inference requests and corresponding responses  |
+----------------+-------------------------------------------------+
| feedback.log   | Inference feedback and corresponding requests   |
+----------------+-------------------------------------------------+

1. Basic Logging Configuration
------------------------------

BentoML supports basic logging configuring under the :code:`logging` section of the configuration.

.. code-block:: yaml
    :caption: BentoML Configuration

    logging:
        level: INFO
        console:
            enabled: True
        file:
            enabled: True
            directory: /var/logs/directory

Refer to the :ref:`configuration guide <configuration-page>` on how override configuration properties.

2. Advanced Logging Configuration
---------------------------------

BentoML uses the standard `Python logging <https://docs.python.org/3/howto/logging.html>`_
module and provides logging customization through advanced :code:`logging` section in the configuration. 
To use advanced logging, set :code:`logging/advanced/enabled` to :code:`True` and provide the standard 
Python logging configuration in :code:`dict` format under the :code:`logging/advanced/config` section.

Please see below configuration examples of different logging scenarios in YAML format.

.. code-block:: yaml
    :caption: Enable INFO+ console logging but only WARN+ file logging.

    logging:
        advanced:
            enabled: True
            config:
                version: 1
                disable_existing_loggers: False
                formatters:
                    console:
                        format: '[%(asctime)s] %(levelname)s - %(message)s'
                    dev: 
                        format: '[%(asctime)s] {{%(filename)s:%(lineno)d}} %(levelname)s - %(message)s'
                handlers:
                    console:
                        level: INFO
                        formatter: console
                        class: logging.StreamHandler
                        stream: ext://sys.stdout
                    local:
                        level: WARN
                        formatter: dev
                        class: logging.handlers.RotatingFileHandler
                        filename: '/var/logs/bentoml/active.log'
                        maxBytes: 104857600
                        backupCount: 2
                loggers:
                    bentoml:
                        handlers: [console, local]
                        level: INFO
                        propagate: False
                    bentoml.prediction:
                        handlers: [console]
                        level: INFO
                        propagate: False
                    bentoml.feedback:
                        handlers: [console]
                        level: INFO
                        propagate: False

.. code-block:: yaml
    :caption: Disable all logging except prediction and feedback file logging.

    logging:
        advanced:
            enabled: True
            config:
                version: 1
                disable_existing_loggers: False
                formatters:
                    prediction:
                        (): pythonjsonlogger.jsonlogger.JsonFormatter
                    feedback:
                        (): pythonjsonlogger.jsonlogger.JsonFormatter
                handlers:
                    prediction:
                        class: logging.handlers.RotatingFileHandler
                        formatter: prediction
                        level: INFO
                        filename: '/var/logs/bentoml/prediction.log'
                        maxBytes: 104857600
                        backupCount: 10
                    feedback:
                        class: logging.handlers.RotatingFileHandler
                        formatter: feedback
                        level: INFO
                        filename: '/var/logs/bentoml/feedback.log'
                        maxBytes: 104857600
                        backupCount: 10
                loggers:
                    bentoml:
                        handlers: []
                        level: INFO
                        propagate: False
                    bentoml.prediction:
                        handlers: [prediction]
                        level: INFO
                        propagate: False
                    bentoml.feedback:
                        handlers: [feedback]
                        level: INFO
                        propagate: False


.. code-block:: yaml
    :caption: Default logging configuration.

    logging:
        advanced:
            enabled: True
            config:
                version: 1
                disable_existing_loggers: False
                formatters:
                    console:
                        format: '[%(asctime)s] %(levelname)s - %(message)s'
                    dev: 
                        format: '[%(asctime)s] {{%(filename)s:%(lineno)d}} %(levelname)s - %(message)s'
                    prediction:
                        (): pythonjsonlogger.jsonlogger.JsonFormatter
                    feedback:
                        (): pythonjsonlogger.jsonlogger.JsonFormatter
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
                        filename: '/var/logs/bentoml/active.log'
                        maxBytes: 104857600
                        backupCount: 2
                    prediction:
                        class: logging.handlers.RotatingFileHandler
                        formatter: prediction
                        level: INFO
                        filename: '/var/logs/bentoml/prediction.log'
                        maxBytes: 104857600
                        backupCount: 10
                    feedback:
                        class: logging.handlers.RotatingFileHandler
                        formatter: feedback
                        level: INFO
                        filename: '/var/logs/bentoml/feedback.log'
                        maxBytes: 104857600
                        backupCount: 10
                loggers:
                    bentoml:
                        handlers: [console, local]
                        level: INFO
                        propagate: False
                    bentoml.prediction:
                        handlers: [console, prediction]
                        level: INFO
                        propagate: False
                    bentoml.feedback:
                        handlers: [console, feedback]
                        level: INFO
                        propagate: False

Refer to the :ref:`configuration guide <configuration-page>` on how override configuration properties.


.. spelling::

    opentracing

