====================
Bento and model APIs
====================

This page contains management APIs for Bentos and models.

Manage Bentos
-------------

.. autofunction:: bentoml.list
.. autofunction:: bentoml.get
.. autofunction:: bentoml.delete
.. autofunction:: bentoml.export_bento
.. autofunction:: bentoml.import_bento
.. autofunction:: bentoml.build
.. autofunction:: bentoml.bentos.build_bentofile

Load models
-----------

.. autoclass:: bentoml.models.BentoModel
    :members: to_info, from_info, resolve
    :undoc-members:
    :show-inheritance:

.. autoclass:: bentoml.models.HuggingFaceModel
    :members: to_info, from_info, resolve
    :undoc-members:
    :show-inheritance:

Manage models
-------------

.. autofunction:: bentoml.models.create
.. autofunction:: bentoml.models.list
.. autofunction:: bentoml.models.get
.. autofunction:: bentoml.models.delete
.. autofunction:: bentoml.models.export_model
.. autofunction:: bentoml.models.import_model


Work with BentoCloud
--------------------

.. autofunction:: bentoml.push
.. autofunction:: bentoml.pull
.. autofunction:: bentoml.models.push
.. autofunction:: bentoml.models.pull
