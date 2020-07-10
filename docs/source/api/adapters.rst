.. _bentoml-api-adapters-label:

API InputAdapters(former Handlers)
==================================

DataframeInput
++++++++++++++
.. autoclass:: bentoml.adapters.DataframeInput

JsonInput
+++++++++
.. autoclass:: bentoml.adapters.JsonInput

TfTensorInput
+++++++++++++
.. autoclass:: bentoml.adapters.TfTensorInput

ImageInput
++++++++++
.. autoclass:: bentoml.adapters.ImageInput

MultiImageInput
+++++++++++++++
.. autoclass:: bentoml.adapters.MultiImageInput

LegacyImageInput
++++++++++++++++
.. autoclass:: bentoml.adapters.LegacyImageInput

FastaiImageInput
++++++++++++++++
.. autoclass:: bentoml.adapters.FastaiImageInput

ClipperInput
++++++++++++

A special group of adapters that are designed to be used when deploying with `Clipper <http://clipper.ai/>`_.

.. autoclass:: bentoml.adapters.ClipperBytesInput
.. autoclass:: bentoml.adapters.ClipperFloatsInput
.. autoclass:: bentoml.adapters.ClipperIntsInput
.. autoclass:: bentoml.adapters.ClipperDoublesInput
.. autoclass:: bentoml.adapters.ClipperStringsInput
