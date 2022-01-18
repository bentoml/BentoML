.. _api-reference-page:


API Reference
=============

This page contains detailed API specification generated from docstring in BentoML
source code. This is best use for digging deeper into BentoML implementation when
learning about advanced features or debugging issues.


BentoService
------------

.. autoclass:: bentoml.Service

.. autofunction:: bentoml.load

bentoml.build
---

.. autofunction:: bentoml.build

Model Store
-----------

.. automodule:: bentoml.models
    :members:

Bento Store
-----------

.. autofunction:: bentoml.list
.. autofunction:: bentoml.get
.. autofunction:: bentoml.delete
.. autofunction:: bentoml.export_bento
.. autofunction:: bentoml.import_bento


Runner
------

.. autoclass:: bentoml.Runner

.. autoclass:: bentoml.SimpleRunner


Tag
---

.. autoclass:: bentoml._internal.types.Tag


API IO Descriptors
------------------

.. automodule:: bentoml.io
    :members: