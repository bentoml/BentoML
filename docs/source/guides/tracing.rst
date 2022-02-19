.. _tracing-page:

Performance Tracing
===================

BentoML API server supports tracing with both `Zipkin <https://zipkin.io/>`_ and
`Jaeger <https://www.jaegertracing.io/>`_. To config tracing server, user can provide a
config YAML file specifying the tracer type and tracing server information:

.. code-block:: yaml

    tracing:
      type: jaeger
      zipkin:
        url: http://localhost:9411/api/v2/spans
      jaeger:
        address: localhost
        port: 6831

Here's an example config for tracing with a Zipkin server:

.. code-block:: yaml

    tracing:
       type: zipkin
       zipkin:
         url: http://localhost:9411/api/v2/spans

When using Zipkin tracer, BentoML only supports its v2 protocol. If you are reporting to
the an OpenZipkin server directly, make sure to add the URL path :code:`/api/v2/spans`
to the server address.

Here is another example config file for tracing with Jaeger and opentracing:

.. code-block:: yaml

    tracing:
      type: jaeger
      jaeger:
        address: localhost
        port: 6831


When starting a BentoML API model server, provide the path to this config file via the
CLI argument `--config`:

.. code-block:: bash

    bentoml serve $BENTO_BUNDLE_PATH --config my_config_file.yml

After BentoML v0.13.0, user will need to provide the config file path via environment
variable :code:`BENTOML_CONFIG`:

.. code-block:: bash

    BENTOML_CONFIG=my_config_file.yml bentoml serve $BENTO_BUNDLE_PATH


Similarly when serving with BentoML API server docker image, assuming you have a
:code:`my_config_file.yml` file ready in current directory:

.. code-block:: bash

    docker run -v $(PWD):/tmp my-bento-api-server -p 3000:3000 --config /tmp/my_config_file.yml

    # after version 0.13.0
    docker run -v $(PWD):/tmp -p 3000:3000 -e BENTOML_CONFIG=/tmp/my_config_file.yml my-bento-api-server

.. spelling::

    opentracing
