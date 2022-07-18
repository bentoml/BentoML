=================
Performance Gudie
=================

This guide is intended to aid advanced BentoML users in better understanding the costs
and performance of their model serving workloads by providing insights into BentoML's
architecture and how to fine tune its performance.

.. TODO::

    Performance Guide Todo items:

    * basic load testing with locust
    * load testing tips:
        * the use of --production
        * enable/disable logging
        * always run locust client on a separate machine

    * performance best practices:
        * ``bentoml serve`` options: --api-worker, --backlog, --timeout
        * configure runner resources
        * configure adaptive batching (max_latency, max_batch_size)

    * existing benchmark results and comparisons

    * advanced topics:
        * alternative load testing with grafana k6
        * setup tracing and dashboard
        * setup tracing for Yatai and distributed Runner
        * instrument tracing for user service and runner code

.. admonition:: Help us improve the project!

    Found an issue or a TODO item? You're always welcome to make contributions to the
    project and its documentation. Check out the
    `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.

