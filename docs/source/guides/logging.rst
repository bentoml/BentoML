=======
Logging
=======

*time expected: 6 minutes*

Server Logging
--------------

BentoML provides a powerful and detailed logging pattern out of the box. Request logs for
webservices are logged along with requests to each of the model runner services.

The request log format is as follows:

.. parsed-literal::
 
    time [LEVEL] [component] ClientIP:ClientPort (scheme,method,path,type,length) (status,type,length) Latency (trace,span,sampled)

For example, a log message might look like:

.. parsed-literal::

    2022-06-28T18:07:35-0700 [INFO] [api_server] 127.0.0.1:37386 (scheme=http,method=POST,path=/classify,type=application/json,length=20) (status=200,type=application/json,length=3) 0.005ms (trace=67131233608323295915755120473254509377,span=4151694932783368069,sampled=0)


OpenTelemetry Compatible
^^^^^^^^^^^^^^^^^^^^^^^^

The BentoML logging system implements the `OpenTelemetry <https://opentelemetry.io/docs/>`_ standard
for :github:`HTTP <open-telemetry/opentelemetry-specification/blob/main/specification/trace/semantic_conventions/http.md>`
throughout the call stack to provide for maximum debuggability. Propogation of the OpenTelemetry
parameters follows the standard provided `here <https://opentelemetry.lightstep.com/core-concepts/context-propagation/>`_

The following are parameters which are provided in the logs as well for correlation back to
particular requests.

- `trace_id` is the id of a trace which tracks “the progression of a single request, as it is handled by services that make up an application.” [#basic_documentation]_

- `span_id is` the id of a span which is contained within a trace.

  .. epigraph::
     “A span is the building block of a trace and is a named, timed operation that represents a piece of the workflow in the distributed system. Multiple spans are pieced together to create a trace.” [#span_documentation]_

- `sampled_id is` the number of times this trace has been sampled.

  .. epigraph::
     “Sampling is a mechanism to control the noise and overhead introduced by OpenTelemetry by reducing the number of samples of traces collected and sent to the backend.” [#sampling_documentation]_

Logging Configuration
^^^^^^^^^^^^^^^^^^^^^

Access logs can be configured by setting the appropriate flags in the bento configuration file for
both web requests and model serving requests. Read more about how to use a bento configuration file
here in the - :doc:`Configuration Guide </guides/configuration>`

To configure other logs, please use the `default Python logging configuration <https://docs.python.org/3/howto/logging.html>`_. All BentoML logs are logged under the ``bentoml`` namespace.

API Service Request Logging
"""""""""""""""""""""""""""

For requests made to the API server, logging can be enabled and disabled using the ``api_server.logging.access`` parameter at the
top level of the ``bentoml_configuration.yml``.

.. code-block:: yaml

    api_server:
      logging:
        access:
          enabled: true
          # whether to log the size of the request body
          request_content_length: true
          # whether to log the content type of the request
          request_content_type: true
          # whether to log the content length of the response
          response_content_length: true
          # whether to log the content type of the response
          response_content_type: true


Model Server Request Logging
""""""""""""""""""""""""""""

Depending on how you've configured BentoML, the API server and runner server can be run
separately. In either case, BentoML also provides a logging configuration under
``runners`` to allow user to configure the runner server output logs.

You may configure the runner access logs under the runners parameter at the top level of your ``bentoml_configuration.yml``:

.. code-block:: yaml

    runners:
      logging:
        access:
          enabled: true
          ...

The available configuration options are the same to the one for API service request logging options above.

Access Logging Format
"""""""""""""""""""""

You may configure the format of the Trace and Span IDs in the access logs in ``bentoml_configuration.yml``.

The default configuration is shown below, where the opentelemetry ``trace_id`` and ``span_id`` are logged in
hexadecimal format, consistent with OpenTelemetry logging instrumentation. You may also configure other format
specs, such as decimal ``d``.

.. code-block:: yaml

    api_server:
      logging:
        access:
          format:
            trace_id: 032x
            span_id: 016x


Library Logging
---------------

When using BentoML as a library, BentoML does not configure any logs. By default, Python will configure a root logger that logs at level WARNING and higher. If you want to see BentoML's DEBUG or INFO logs, register a log handler to the ``bentoml`` namespace:

.. code-block:: python

    import logging

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    bentoml_logger = logging.getLogger("bentoml")
    bentoml_logger.addHandler(ch)
    bentoml_logger.setLevel(logging.DEBUG)

----

.. rubric:: Notes

.. [#basic_documentation] `OpenTelemetry Basic Documentation <https://www.dynatrace.com/support/help/extend-dynatrace/opentelemetry/basics>`_

.. [#span_documentation] `OpenTelemetry Span Documentation <https://opentelemetry.lightstep.com/spans/>`_

.. [#sampling_documentation] `OpenTelemetry SDK Documentation <https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/sdk.md>`_

