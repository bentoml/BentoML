.. _bentoml-api-handlers-label:

API InputAdapters(former Handlers)
================

DataframeInput
++++++++++++++++
.. autoclass:: bentoml.adapters.DataframeInput

JsonHandler
+++++++++++
.. autoclass:: bentoml.handlers.JsonHandler

TensorflowTensorHandler
+++++++++++++++++++++++
.. autoclass:: bentoml.handlers.TensorflowTensorHandler

ImageHandler
++++++++++++
.. autoclass:: bentoml.handlers.ImageHandler

LegacyImageHandler
++++++++++++
.. autoclass:: bentoml.handlers.LegacyImageHandler

FastaiImageHandler
++++++++++++++++++
.. autoclass:: bentoml.handlers.FastaiImageHandler

ClipperInput
++++++++++++++

A special group of handlers that are designed to be used when deploying with `Clipper <http://clipper.ai/>`_.

.. autoclass:: bentoml.handlers.ClipperBytesHandler
.. autoclass:: bentoml.handlers.ClipperFloatsHandler
.. autoclass:: bentoml.handlers.ClipperIntsHandler
.. autoclass:: bentoml.handlers.ClipperDoublesHandler
.. autoclass:: bentoml.handlers.ClipperStringsHandler
