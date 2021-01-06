.. _configuration-page:

Configuration
=============

BentoML can be configured through configuration properties defined in the `default_bentoml.cfg <https://github.com/bentoml/BentoML/blob/master/bentoml/configuration/default_bentoml.cfg>`_. 
The values of configuration properties are applied in the following precedence order. 

- Environment Variables
- User Defined Configuration File
- BentoML Defaults

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

To override a configuration property, environment variables should be named in the following convention, 
`BENTOML__<SECTION>__<KEY>`, in upper case letters.

For example, to override the `level` property to `ERROR` in the `logging` section of the configuration, user 
should define an environment variable named `BENTOML__LOGGING__LEVEL` with value `ERROR`.


.. code-block:: cfg
    :caption: default_bentoml.cfg

    [logging]
    level = INFO

See Docker example below for setting the environment variable of logging level.

.. code-block:: shell

    $ docker run -e BENTOML__LOGGING__LEVEL=ERROR

User Defined Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A user defined configuration file, in the same format as the 
`default_bentoml.cfg <https://github.com/bentoml/BentoML/blob/master/bentoml/configuration/default_bentoml.cfg>`_ 
can be placed under the BentoML home directory with the file name `bentoml.cfg`, to override existing configuration 
properties.

The example below, overrides both `level` and `file_logging_enabled` properties in the `logging` section, to change 
logging level to `WARN` and disable file based logging.

.. code-block:: cfg
    :caption: {BENTOML_HOME}/bentoml.cfg

    [logging]
    level = WARN
    file_logging_enabled = false

See Docker example below for injecting the BentoML configuration file into the container.

.. code-block:: shell

    $ docker run -v /local/path/to/bentoml.cfg:{BENTOML_HOME}/bentoml.cfg

BentoML Defaults
^^^^^^^^^^^^^^^^

Any non-overridden properties will fallback to the default values defined in 
`default_bentoml.cfg <https://github.com/bentoml/BentoML/blob/master/bentoml/configuration/default_bentoml.cfg>`_. 