=========================
Custom Metrics in BentoML
=========================
Metrics are quantitative assessments for measuring and conducting assessments of your system's performance. Metrics are vital in providing reliable and accurate diagnoses to your BentoService in real-time scenarios. BentoML enables users to define custom metrics with `Prometheus <https://prometheus.io/>`_ to bootstrap and fast-track one's monitoring story. 
 
This article will dive into how to add custom metrics to monitor your BentoService and a quick tutorial on how one can incorporate custom metrics to a :ref:`concepts/runner:Custom Runner`.

Be sure to review the :ref:`reference/metrics_methods:Metrics in BentoML` to learn more. 

Using Metrics with :ref:`concepts/runner:Custom Runner`
-------------------------------------------------------
.. TODO: Add example



Using Metrics with BentoService
-------------------------------
.. literalinclude:: ../../../examples/bentoml_metrics/services_metrics/service.py
   :language: python
   :caption: `service.py`

After a metric has been created and added to a service. Users can go to ``/metrics`` or  port ``3001``, depending on if gRPC or HTTP is being monitored, to access metrics.