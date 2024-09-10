=======
Logging
=======

BentoML provides a built-in logging system to provide comprehensive insights into the operation of your BentoML Services. It implements the `OpenTelemetry <https://opentelemetry.io/docs/>`_ standard to propagate critical information throughout the HTTP call stack for detailed debugging and analysis.

This document provides guidance on configuring logging in BentoML, including managing server-side logs and customizing logging for BentoML as a library.

Server logging
--------------

Server logging is enabled by default in BentoML. After a Service starts, every request made to the server is logged with detailed information. BentoML structures the logs to provide a clear and concise overview of each request, formatted as follows:


.. code-block:: bash

	timestamp [LEVEL] [component] ClientIP:ClientPort (scheme,method,path,type,length) (status,type,length) Latency (trace,span,sampled,service.name)

An example of a log message for a request processed by BentoML might look like this:

.. code-block:: bash

	2024-04-13T02:03:49+0000 [INFO] [entry_service:Summarization:1] 44.xxx.xxx.104:7181 (scheme=http,method=GET,path=/docs.json,type=,length=) (status=200,type=application/json,length=5543) 1.972ms (trace=7589d361df3e8ad3f0a71acb44d150be,span=07ef3bc1685d067c,sampled=0,service.name=Summarization)

This log entry provides detailed information about the request, including the client IP and port, request method, path, payload type and length, response status, response content type and length, request latency, and OpenTelemetry identifiers.

BentoML's logging system is fully compatible with the OpenTelemetry standard. The server log contains several OpenTelemetry parameters that are useful for correlating logs back to specific requests or operations.

- ``trace``: Identifies a trace, which consists of one or multiple spans that represent a single request flowing through multiple Services. See `Traces <https://opentelemetry.io/docs/concepts/signals/traces/>`_ for details.
- ``span``: Identifies an individual span within a trace. Each span represents a specific operation or a unit of work within the trace, such as a single HTTP request. See `Spaces <https://opentelemetry.io/docs/concepts/signals/traces/#spans>`_ for details.
- ``sampled``: Indicates whether a trace is being sampled, namely whether or not the trace data should be recorded. If sampling is enabled (usually denoted as ``1`` for sampled and ``0`` for not sampled), only a subset of traces is captured, which helps manage data volume and reduce performance overhead. See `Sampling <https://opentelemetry.io/docs/concepts/sampling/>`_ for details.

Configure logging
^^^^^^^^^^^^^^^^^

You can configure server logging in your Service definition by using the ``logging`` parameter in the ``@bentoml.service`` decorator.

.. code-block:: python

	import bentoml

	@bentoml.service(logging={
	    "access": {
	        "enabled": True,
	        "request_content_length": True,
	        "request_content_type": True,
	        "response_content_length": True,
	        "response_content_type": True,
	        "skip_paths": ["/metrics", "/healthz", "/livez", "/readyz"],
	        "format": {
	            "trace_id": "032x",
	            "span_id": "016x"
	        }
	    }
	})
	class MyService:
	    # Service implementation

Available logging parameters to provide control over what data is logged and how it is formatted:

- ``enabled``: Enables or disables logging.
- ``request_content_length``: Logs the size of the request body.
- ``request_content_type``: Logs the content type of the request.
- ``response_content_length``: Logs the size of the response body.
- ``response_content_type``: Logs the content type of the response.
- ``skip_paths``: Specifies route paths that should be excluded from logging.
- ``format``: Customizes the logging format of OpenTelemetry trace identifiers.

  - ``trace_id``: Logs the trace identifier in a specified format, such as ``032x``.
  - ``span_id``: Logs the span identifier in a specified format, such as ``016x``.

To configure other logs, use the `default Python logging configuration <https://docs.python.org/3/howto/logging.html>`_. All BentoML logs are logged under the ``bentoml`` namespace.

Library logging
---------------

When you use BentoML as a library for your Python application, it does not configure any logs, without any specific handlers, formatters, or filters. This means that without additional configuration, the logging output from BentoML would follow the Python root logger's settings, which by default logs messages at the WARNING level and higher (including ERROR and CRITICAL).

To capture more detailed logs from BentoML, especially at the ``DEBUG`` or ``INFO`` levels, you must explicitly set up and register a log handler to the ``bentoml`` namespace. Here is a simple example of how to do this:

.. code-block:: python

	import logging

	# Create a stream handler
	ch = logging.StreamHandler()

	# Set a format for the handler
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)

	# Get the BentoML logger
	bentoml_logger = logging.getLogger("bentoml")

	# Add the handler to the BentoML logger
	bentoml_logger.addHandler(ch)

	# Set the desired logging level (e.g., DEBUG)
	bentoml_logger.setLevel(logging.DEBUG)

.. note::

	When starting a Service using ``bentoml serve``, the command forks ``service.py`` into a child process. Handlers that involve file operations, such as log rotation (``RotatingFileHandler`` or ``TimedRotatingFileHandler``), are not supported within the Service definition. For more information, see `the Python Logging Cookbook <https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes>`_.
