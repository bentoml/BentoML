===================
Configuring BentoML
===================

BentoML starts with an out-of-the-box configuration that works for common use cases. For advanced users, many
features can be customized through configuration. Both BentoML CLI and Python APIs can be customized 
by the configuration. Configuration is best used for scenarios where the customizations can be specified once 
and applied to the entire team.

BentoML configuration is defined by a YAML file placed in a directory specified by the ``BENTOML_CONFIG`` 
environment variable. The example below starts the bento server with configuration defined in ``~/bentoml_configuration.yaml``:

.. code-block:: shell

    $ BENTOML_CONFIG=~/bentoml_configuration.yaml bentoml serve iris_classifier:latest

Users only need to specify a partial configuration with only the properties they wish to customize instead 
of a full configuration schema. In the example below, the microbatching workers count is overridden to 4.
Remaining properties will take their defaults values.

.. code-block:: yaml
   :caption: `~/bentoml_configuration.yaml`

    api_server:
      workers: 4
      timeout: 60
      http:
        port: 6000

Throughout the BentoML documentation, features that are customizable through configuration are demonstrated 
like the example above. For a full configuration schema including all customizable properties, refer to
the BentoML configuration template defined in :github:`default_configuration.yml <bentoml/BentoML/blob/main/bentoml/_internal/configuration/default_configuration.yaml>`.



Docker Deployment
-----------------

Configuration file can be mounted to the Docker container using the `-v` option and specified to the BentoML 
runtime using the `-e` environment variable option.

.. code-block:: shell

    $ docker run -v /local/path/configuration.yml:/home/bentoml/configuration.yml -e BENTOML_CONFIG=/home/bentoml/configuration.yml


.. spelling::

    customizations
    microbatching
    customizable
    multiproc
    dir
    tls
    apiserver
    uri
    gcs
