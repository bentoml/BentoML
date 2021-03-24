Logging
=======

BentoML API server logs all the prediction requests and their responses under the
:code:`$BENTOML_HOME/logs` directory by default in a local development environment. When
deploying with Docker, the logs can be found under the :code:`/home/bentoml/logs`
directory.

BentoML uses standard `Python logging <https://docs.python.org/3/howto/logging.html>`_
module and provides logging customization through :code:`logging` section in :code:`bentoml.cfg`
under the BentoML home directory. Refer to the :ref:`configuration guide <configuration-page>` on how override configuration properties. See
`default_bentoml.cfg <https://github.com/bentoml/BentoML/blob/master/bentoml/configuration/default_bentoml.cfg>`_
for a list of override-able properties.

For advanced logging customization, user can provide full logging configurations in :code:`logging.yml`, placed under
the BentoML home directory. For example, :code:`logging.yml` configuration file can be injected into the Docker container
through the following command:

.. code-block:: shell

    $ docker run -v /local/path/to/logging.yml:/home/bentoml/logging.yml

Please see below configuration examples of different logging scenarios in YAML format.

.. code-block:: yaml
    :caption: Enable INFO+ console logging but only WARN+ file logging.

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
            filename: '/var/log/bentoml/active.log'
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
            filename: '/var/log/bentoml/prediction.log'
            maxBytes: 104857600
            backupCount: 10
        feedback:
            class: logging.handlers.RotatingFileHandler
            formatter: feedback
            level: INFO
            filename: '/var/log/bentoml/feedback.log'
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
            filename: '/var/log/bentoml/active.log'
            maxBytes: 104857600
            backupCount: 2
        prediction:
            class: logging.handlers.RotatingFileHandler
            formatter: prediction
            level: INFO
            filename: '/var/log/bentoml/prediction.log'
            maxBytes: 104857600
            backupCount: 10
        feedback:
            class: logging.handlers.RotatingFileHandler
            formatter: feedback
            level: INFO
            filename: '/var/log/bentoml/feedback.log'
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
