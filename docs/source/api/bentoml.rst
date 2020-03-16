BentoService API
================

.. _bentoml-bentoservice-label:

BentoService
++++++++++++

.. autoclass:: bentoml.BentoService

  .. automethod:: bentoml.BentoService.name

  .. automethod:: bentoml.BentoService.versioneer

  .. automethod:: bentoml.BentoService.set_version

  .. automethod:: bentoml.BentoService.get_service_apis

  .. method:: bentoml.BentoService.artifacts

        returns a dictionary of packed artifacts from the artifact name to the artifact
        model instance in its native form

  .. method:: pack(name, *args, *kwargs)

        BentoService#pack method is used for packing trained model instances with a
        BentoService instance and make it ready for BentoService#save.

        :param name: name of the declared model artifact
        :param args: args passing to the target model artifact to be packed
        :param kwargs: kwargs passing to the target model artifact to be packed
        :return: this BentoService instance

  .. automethod:: bentoml.BentoService.pack




api
+++
.. autofunction:: bentoml.api

.. _bentoml-env-label:

env
+++
.. autofunction:: bentoml.env

artifacts
+++++++++
.. autofunction:: bentoml.artifacts

ver
+++
.. autofunction:: bentoml.ver

save
++++
.. autofunction:: bentoml.save

save_to_dir
+++++++++++
.. autofunction:: bentoml.save_to_dir

load
++++
.. autofunction:: bentoml.load
