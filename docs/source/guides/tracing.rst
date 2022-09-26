=======
Tracing
=======

BentoML API server supports tracing with `Zipkin <https://zipkin.io/>`_,
`Jaeger <https://www.jaegertracing.io/>`_ and `OTLP <https://opentelemetry.io/>`_.

:bdg-info:`Requirements:` bentoml must be installed with the extras dependencies for
tracing exporters. The following command will install BentoML with its coresponding
tracing exporter:

.. tab-set::

   .. tab-item:: Jaeger

       .. code-block:: bash

          pip install "bentoml[tracing-jaeger]"

   .. tab-item:: Zipkin

      .. code-block:: bash

         pip install "bentoml[tracing-zipkin]"

   .. tab-item:: OpenTelemetry Protocol

      .. code-block:: bash

         pip install "bentoml[tracing-otlp]"

To config tracing server, user can provide a config YAML file specifying the tracer type and tracing server information:

.. code-block:: yaml

    tracing:
      type: jaeger
      sample_rate: 1.0
      zipkin:
        url: http://localhost:9411/api/v2/spans
      jaeger:
        address: localhost
        port: 6831
      otlp:
        protocol: grpc
        url: http://localhost:4317

By default, no traces will be collected. Set sample_rate to your desired fraction in order to start collecting them.
Here is an example config for tracing with a Zipkin server:

.. code-block:: yaml

    tracing:
       type: zipkin
       sample_rate: 1.0
       zipkin:
         url: http://localhost:9411/api/v2/spans

When using Zipkin tracer, BentoML only supports its v2 protocol. If you are reporting to
the an OpenZipkin server directly, make sure to add the URL path :code:`/api/v2/spans`
to the server address.

Here is another example config file for tracing with Jaeger and opentracing:

.. code-block:: yaml

    tracing:
      type: jaeger
      sample_rate: 1.0
      jaeger:
        address: localhost
        port: 6831

If you would like to exclude some routes from tracing, you can specify them using
the :code:`excluded_urls` parameter. This parameter can be either a comma-separated 
string of routes, or a list of strings.

.. code-block:: yaml

    tracing:
      type: jaeger
      sample_rate: 1.0
      jaeger:
        address: localhost
        port: 6831
      excluded_urls: readyz,livez,healthz,static_content,docs,metrics


Finally, here is an example using OTLP. This allows easy integration with an OpenTelemetry Traces receiver.
You may use either HTTP or gRPC as protocol. gRPC is the default, but HTTP may be easier to proxy or load-balance.

.. code-block:: yaml

    tracing:
      type: otlp
      sample_rate: 1.0
      otlp:
        protocol: grpc
        url: http://localhost:4317

If using HTTP, you must set the whole Traces receiver endpoint path (e.g. `/v1/traces` for OpenTelemetry Collector):

.. code-block:: yaml

    tracing:
      type: otlp
      sample_rate: 1.0
      otlp:
        protocol: http
        url: http://localhost:4318/v1/traces

When starting a BentoML API model server, provide the path to this config file
by setting the environment variable :code:`BENTOML_CONFIG`:

.. code-block:: bash

    BENTOML_CONFIG=my_config_file.yml bentoml serve $BENTO_BUNDLE_PATH


Similarly when serving with BentoML API server docker image, assuming you have a
:code:`my_config_file.yml` file ready in current directory:

.. code-block:: bash

    docker run -v $(PWD):/tmp -p 3000:3000 -e BENTOML_CONFIG=/tmp/my_config_file.yml my-bento-api-server

.. spelling::

    opentracing
