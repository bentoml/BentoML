=============
Configuration
=============

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
the BentoML configuration template defined in :github:`default_configuration.yml <bentoml/BentoML/blob/main/src/bentoml/_internal/configuration/default_configuration.yaml>`.




Overrding configuration with environment variables
--------------------------------------------------

Users can also override configuration fields with environment variables. by defining
an oneline value of a "flat" JSON via ``BENTOML_CONFIG_OPTIONS``:

.. code-block:: yaml

   $ BENTOML_CONFIG_OPTIONS='runners.pytorch_mnist.resources."nvidia.com/gpu"[0]=0 runners.pytorch_mnist.resources."nvidia.com/gpu"[1]=2' \
            bentoml serve pytorch_mnist_demo:latest --production

Which the override configuration will be intepreted as:

.. code-block:: yaml

   runners:
    pytorch_mnist:
      resources:
        nvidia.com/gpu: [0, 2]

.. note::

   For fields that represents a iterable type, the override array must have a space
   separating each element:

   .. image:: /_static/img/configuration-override-env.png
      :alt: Configuration override environment variable


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
