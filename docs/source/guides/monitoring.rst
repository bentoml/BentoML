Monitoring with Prometheus
==========================

Monitoring stacks usually consist of a metrics collector, a time-series database to store metrics
and a visualization layer. A popular stack is `Prometheus <https://prometheus.io/>`_ with `Grafana <https://grafana.com/>`_
used as the visualization layer to create rich dashboards. An architecture of Prometheus and its ecosystem is shown below:

.. image:: ../_static/img/prom-architecture.png


BentoML API server comes with Prometheus support out of the box. When launching an API model server with BentoML,
whether it is running dev server locally or deployed with docker in the cloud, a ``/metrics`` endpoint will always
be available for exposing prometheus metrics.

.. note::
    Currently custom metrics is not yet supported in current version of BentoML.

We are working on more documentation around setting up a grafana 
dashboard for monitoring BentoML API model server, adding custom metrics
and other advanced usages for monitoring.