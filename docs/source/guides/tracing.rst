=======
Tracing
=======

*time expected: 12 minutes*

This guide dives into the :wiki:`tracing <Tracing_(software)>` capabilities that BentoML offers.

BentoML allows user to export trace with `Zipkin <https://zipkin.io/>`_,
`Jaeger <https://www.jaegertracing.io/>`_ and `OTLP <https://opentelemetry.io/>`_.

This guide will also provide a simple example of how to use BentoML tracing with `Jaeger <https://www.jaegertracing.io/>`_

Why do you need this?
---------------------

Debugging models and services in production is hard. Adding logs and identifying
the root cause of the problem is time consuming and error prone. Additionally, tracking
logs across multiple services is difficult, which takes a lot of time, and slow down
your development agility. As a result, logs won’t always provide the required information to solve regressions.

Tracing encompasses a much wider, continuous view of an application. The goal of tracing is to following a program’s flow and data progression.
As such, there is a lot more information at play; tracing can be a lot noisier than logging – and that’s intentional.

BentoML comes with built-in tracing support, with :ref:`OpenTelemetry <guides/logging:OpenTelemetry Compatible>`. This means users
can then use any of the OpenTelemetry compatible tracing tools to visualize and analyze the traces.

Running a BentoService
----------------------

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

We will be using the example from :ref:`the quickstart <tutorial:Tutorial: Intro to BentoML>`.

Run the Jaeger `all-in-one <https://www.jaegertracing.io/docs/1.38/getting-started/#all-in-one>`_ docker image:

.. code-block:: bash

   » docker run -d --name jaeger \
      -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
      -e COLLECTOR_OTLP_ENABLED=true \
      -p 6831:6831/udp \
      -p 6832:6832/udp \
      -p 5778:5778 \
      -p 16686:16686 \
      -p 4317:4317 \
      -p 4318:4318 \
      -p 14250:14250 \
      -p 14268:14268 \
      -p 14269:14269 \
      -p 9411:9411 \
      jaegertracing/all-in-one:1.38

.. dropdown:: For our Mac users
   :icon: cpu

   If you are running into this error:

   .. parsed-literal::

      2022-10-05T01:32:21-0700 [WARNING] [api_server:iris_classifier:8] Data exceeds the max UDP packet size; size 216659, max 65000
      2022-10-05T01:32:24-0700 [ERROR] [api_server:iris_classifier:3] Exception while exporting Span batch.
      Traceback (most recent call last):
        File "~/venv/lib/python3.10/site-packages/opentelemetry/sdk/trace/export/__init__.py", line 367, in _export_batch
          self.span_exporter.export(self.spans_list[:idx])  # type: ignore
        File "~/venv/lib/python3.10/site-packages/opentelemetry/exporter/jaeger/thrift/__init__.py", line 219, in export
          self._agent_client.emit(batch)
        File "~/venv/lib/python3.10/site-packages/opentelemetry/exporter/jaeger/thrift/send.py", line 95, in emit
          udp_socket.sendto(buff, self.address)
      OSError: [Errno 40] Message too long

   This is because the default UDP packet size on Mac is set 9216 bytes, which is described `under Jaeger reporters <https://www.jaegertracing.io/docs/1.19/client-libraries/#emsgsize-and-udp-buffer-limits>`_. To increase the UDP packet size, run the following command:

   .. code-block:: bash

      % sysctl net.inet.udp.maxdgram
      # net.inet.udp.maxdgram: 9216
      % sudo sysctl net.inet.udp.maxdgram=65536
      # net.inet.udp.maxdgram: 9216 -> 65536
      % sudo sysctl net.inet.udp.maxdgram
      # net.inet.udp.maxdgram: 65536


To configure Jaeger exporter, user can provide a config :wiki:`YAML` file specifying the tracer type and tracing server information under ``api_server.tracing``:

.. code-block:: yaml
   :caption: `bentoml_configuration.yaml`

    api_server:
      tracing:
        exporter_type: jaeger
        sample_rate: 0.8
        jaeger:
          protocol: thrift
          thrift:
            agent_host_name: localhost
            agent_port: 6831

Run the BentoService with the config file:

.. code-block:: bash

   » BENTOML_CONFIG=bentoml_configuration.yaml bentoml serve iris_classifier:latest --production

Send any request to the BentoService, and then you can visit the `Jaeger UI <http://localhost:16686>`_ to see the traces.

.. image:: /_static/img/jaeger-ui.png
   :alt: Jaeger UI

:raw-html:`<br />`

Exporter Configuration
----------------------

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
