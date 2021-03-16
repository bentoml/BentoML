Performance Tracing
===================

BentoML API server supports tracing with `Zipkin <https://zipkin.io/>`_ or `Opentracing
<https://opentracing.io/>`_. To config tracing server, user can provide a config YAML
file containing either a zipkin server url or an opentracing server address and port:

.. code-block:: yml

    tracing:
      zipkin_api_url: Null
      opentracing_server_address: Null
      opentracing_server_port: Null

When starting a BentoML API model server, provide the CLI command with this YAML config
file with the `--config` option:

.. code-block:: bash

    bentoml serve $BENTO_BUNDLE_PATH --config my_config_file.yml

Similarly when serving with BentoML API server docker image:

.. code-block:: bash

    docker run -v $(PWD):/tmp my-bento-api-server --config /tmp/my_config_file.yml

BentoML has already implemented basic tracing information for its micro-batching server
and model server. If there's additional tracing that you'd like to add to your BentoML
model server, you may import
