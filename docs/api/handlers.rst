API Handlers
================

DataframeHandler
++++++++++++++++
.. autoclass:: bentoml.handlers.DataframeHandler

JsonHandler
+++++++++++
.. autoclass:: bentoml.handlers.JsonHandler

TensorflowTensorHandler
+++++++++++++++++++++++
.. autoclass:: bentoml.handlers.TensorflowTensorHandler

ImageHandler
++++++++++++
.. autoclass:: bentoml.handlers.ImageHandler

FastaiImageHandler
++++++++++++++++++
.. autoclass:: bentoml.handlers.FastaiImageHandler

ClipperHandler
++++++++++++++
A special handler that should only be used when deploying BentoService
 with Clipper(http://clipper.ai/)

.. autoclass:: bentoml.handlers.ClipperBytesHandler
.. autoclass:: bentoml.handlers.ClipperFloatsHandler
.. autoclass:: bentoml.handlers.ClipperIntsHandler
.. autoclass:: bentoml.handlers.ClipperDoublesHandler
.. autoclass:: bentoml.handlers.ClipperStringsHandler