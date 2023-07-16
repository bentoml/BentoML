============================
Resource Scheduling Strategy
============================

To make the best use of machine resources, BentoML provides a few configuration options to control how many runner workers
will be spawned and how they distribute on the CPUs and GPUs. It can be configured for all runners as well as individual runner.
The following configuration options are available:

.. tab-set::

    .. tab-item:: Global configuration (Applies to all runners)
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            resources:
              cpu: 4
              nvidia.com/gpu: 2
            workers_per_resource: 2

    .. tab-item:: Individual Runner
        :sync: individual_runner

        .. code-block:: yaml
           :caption: ⚙️ `configuration.yml`

           runners:
             iris_clf:
               resources:
                 cpu: 4
                 nvidia.com/gpu: 2
               workers_per_resource: 2

Configuration Options
---------------------

``runners.resources.cpu``
    A float value indicating the number of processes(rounded up) to spawn. If the runner supports multi-threading, it will be the number of threads per process.
``runners.resources.nvidia.com/gpu``
    An integer indicating the number of GPUs to use. If the runner supports running on GPU, it will create a number of instances defined by ``runners.workers_per_resource`` on each allocated GPU.
    Alternatively, you can specify what GPUs to use by setting it to a list of GPU IDs, e.g. ``[2, 4]``, the IDs are 1-indexed.
``runners.workers_per_resource``
    A resource multiplier to control how many workers will be spawned on each allocated resource. The default value is 1.

Examples
--------

The following configuration spawns 4 workers on CPU for runners not supporting multi-threading. If the runner supports multi-threading, it will be 4 threads with only one worker.

.. tab-set::

    .. tab-item:: Global configuration (Applies to all runners)
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            resources:
              cpu: 4

    .. tab-item:: Individual Runner
        :sync: individual_runner

        .. code-block:: yaml
           :caption: ⚙️ `configuration.yml`

           runners:
             iris_clf:
               resources:
                 cpu: 4

When running in multi-threading mode, the following configuration will create 2 workers with 4 threads each.
However, if multi-threading is not supported, it will instead create 8 workers by multiplying 2 and 4.

.. tab-set::

    .. tab-item:: Global configuration (Applies to all runners)
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            resources:
              cpu: 4
            workers_per_resource: 2

    .. tab-item:: Individual Runner
        :sync: individual_runner

        .. code-block:: yaml
           :caption: ⚙️ `configuration.yml`

           runners:
             iris_clf:
               resources:
                 cpu: 4
               workers_per_resource: 2

If the runner supports running on GPU, the following configuration will spawn 2 workers on each GPU, hence 4 workers will be spawned for 2 GPUs in total:

.. tab-set::

    .. tab-item:: Global configuration (Applies to all runners)
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            resources:
              nvidia.com/gpu: 2
            workers_per_resource: 2

    .. tab-item:: Individual Runner
        :sync: individual_runner

        .. code-block:: yaml
           :caption: ⚙️ `configuration.yml`

           runners:
             iris_clf:
               resources:
                 nvidia.com/gpu: 2
               workers_per_resource: 2
