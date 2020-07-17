BentoService API
================

.. _bentoml-bentoservice-label:

BentoService
++++++++++++

.. autoclass:: bentoml.BentoService

  .. automethod:: bentoml.BentoService.name

  .. automethod:: bentoml.BentoService.versioneer

  .. automethod:: bentoml.BentoService.set_version

  .. autoattribute:: bentoml.BentoService.inference_apis

  .. autoattribute:: bentoml.BentoService.artifacts


  .. _bentoml-bentoservice-pack-label:

  .. method:: pack(name, *args, *kwargs)

        BentoService#pack method is used for packing trained model instances with a
        BentoService instance and make it ready for BentoService#save.

        :param name: name of the declared model artifact
        :param args: args passing to the target model artifact to be packed
        :param kwargs: kwargs passing to the target model artifact to be packed
        :return: this BentoService instance

  .. automethod:: bentoml.BentoService.pack

  .. _bentoml-bentoservice-save-label:

  .. automethod:: bentoml.BentoService.save


  .. _bentoml-bentoservice-save-to-dir-label:

  .. automethod:: bentoml.BentoService.save_to_dir


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


.. _bentoml-load-label:

load
++++
.. autofunction:: bentoml.load
