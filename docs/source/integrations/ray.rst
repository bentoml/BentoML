===
Ray
===

`Ray <https://docs.ray.io/en/latest/ray-overview/getting-started.html>`_ is a popular open-source compute framework that makes it easy to scale Python workloads. BentoML integrates natively with `Ray Serve <https://docs.ray.io/en/latest/serve/index.html>`_, a library built to scale API services on a Ray cluster, to enable users to deploy Bento applications in a Ray cluster without modifying code or configuration.

The central API in the Ray Serve integration in BentoML is ``bentoml.ray.deployment``, it seamless converts a Bento into a Ray Serve `Deployment <https://docs.ray.io/en/latest/serve/key-concepts.html#deployment>`_. At the simpliest form, only a bento tag is required to create a Deployment.

.. code-block:: python
  :caption: `bento_ray.py`

  import bentoml

  classifier = bentoml.ray.deployment('iris_classifier:latest')

The Ray Serve Deployment can then be deployed locally or to a Ray cluster using the Ray Serve's run command.

.. code-block:: bash

  serve run bento_ray:classifier


Scaling Resources and Autoscaling
---------------------------------

The ``bentoml.ray.deployment`` API also supports configuring `scaling resources and autoscaling behaviors <https://docs.ray.io/en/master/serve/autoscaling-guide.html>`_. In addition to the Bento tag, ``service_deployment_config`` and ``runner_deployment_config`` arguments can be passed in to configure the Deployments of API Server and Runners respectively.
All parameters allowed in Ray Serve Deployment can be specified in the ``service_deployment_config`` and ``runner_deployment_config``. The Runner name should be specified as the key in the ``runner_deployment_config``.

.. code-block:: python
  :caption: `bento_ray.py`

  import bentoml

  classifier = bentoml.ray.deployment(
    'iris_classifier:latest',
    {
        "route_prefix": "/classify",
        "num_replicas": 3,
        "ray_actor_options": {
            "num_cpus": 1
        }
    },
    {
        "iris_clf": {
            "num_replicas": 1,
            "ray_actor_options": {
                "num_cpus": 5
            }
        }
    }
  )

.. note::

    Arguments in the ``service_deployment_config`` and ``runner_deployment_config`` dictionaries are passed through directly to Deployment. Please refer to `Resource Allocation <https://docs.ray.io/en/master/serve/resource-allocation.html>`_ and `Ray Serve Autoscaling <https://docs.ray.io/en/master/serve/autoscaling-guide.html>`_ for the full list of supported arguments.


Batching
--------

`Batching behaviors <https://docs.ray.io/en/latest/serve/tutorials/batch.html>`_ can be configured through the ``enable_batching`` and ``batching_config`` arguments. Using Runner name as the key, both ``max_batch_size`` and ``batch_wait_timeout_s`` can be configured for each Runner independently through ``batching_config``.

.. code-block:: python
  :caption: `bento_ray.py`

  import bentoml

  deploy = bentoml.ray.deployment(
    'fraud_detection:latest',
    enable_batching=True,
    batching_config={
        "iris_clf": {
            "predict": {
                "max_batch_size": 1024,
                "batch_wait_timeout_s": 0.2
            }
        }
    }
  )

.. note::

    Arguments in the ``batching_config`` dictionary are passed through directly to Ray Serve. Please refer to `Ray Serve Batching <https://docs.ray.io/en/latest/serve/tutorials/batch.html>`_ for the full list of supported arguments.


Reference
---------

See the :ref:`API references <reference/frameworks/ray:Ray>` to learn more about the Ray Serve integration in BentoML.
