===================
Configuring BentoML
===================

BentoML starts with an out-of-the-box configuration that works for common use cases. For advanced users, many
features can be customized through configuration. Both BentoML CLI and Python APIs can be customized 
by the configuration. Configuration is best used for scenarios where the customizations can be specified once 
and applied to the entire team.

.. note::
    BentoML configuration underwent a major redesign in release 0.13.0. Previous configuration set through the 
    `bentoml config` CLI command is no longer compatible with the configuration releases in 0.13.0 and above. 
    Please see legacy configuration property mapping table below to upgrade configuration to the new format.

BentoML configuration is defined by a YAML file placed in a directory specified by the :code:`BENTOML_CONFIG` 
environment variable. The example below starts the bento server with configuration defined in :code:`~/bentoml_configuration.yaml`

.. code-block:: shell

    $ BENTOML_CONFIG=~/bentoml_configuration.yml bentoml serve IrisClassifier:latest

Users only need to specify a partial configuration with only the properties they wish to customize instead 
of a full configuration schema. In the example below, the microbatching workers count is overridden to 4. 
Remaining properties will take their defaults values.

.. code-block:: yaml
    :caption: BentoML Configuration

    api_server:
        port: 6000
        timeout: 120
        backlog: 1024

Throughout the BentoML documentation, features that are customizable through configuration are demonstrated 
like the example above. For a full configuration schema including all customizable properties, refer to 
the BentoML configuration template defined in 
`default_configuration.yml <https://github.com/bentoml/BentoML/blob/main/bentoml/_internal/configuration/default_configuration.yaml>`_.


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
