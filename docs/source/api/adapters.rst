.. _bentoml-api-adapters-label:

API InputAdapters
=================

BaseInputAdapter
++++++++++++++++
.. autoclass:: bentoml.adapters.BaseInputAdapter

DataframeInput
++++++++++++++
.. autoclass:: bentoml.adapters.DataframeInput

JsonInput
+++++++++
.. autoclass:: bentoml.adapters.JsonInput

LegacyJsonInput
+++++++++++++++
.. autoclass:: bentoml.adapters.LegacyJsonInput

TfTensorInput
+++++++++++++
.. autoclass:: bentoml.adapters.TfTensorInput

ImageInput
++++++++++
.. autoclass:: bentoml.adapters.ImageInput

MultiImageInput
+++++++++++++++
.. autoclass:: bentoml.adapters.MultiImageInput

AnnotatedImageInput
+++++++++++++++++++
.. autoclass:: bentoml.adapters.AnnotatedImageInput

LegacyImageInput
++++++++++++++++
.. autoclass:: bentoml.adapters.LegacyImageInput

FastaiImageInput
++++++++++++++++
.. autoclass:: bentoml.adapters.FastaiImageInput

FileInput
+++++++++
.. autoclass:: bentoml.adapters.FileInput

ClipperInput
++++++++++++

A special group of adapters that are designed to be used when deploying with `Clipper <http://clipper.ai/>`__.

.. autoclass:: bentoml.adapters.ClipperBytesInput
.. autoclass:: bentoml.adapters.ClipperFloatsInput
.. autoclass:: bentoml.adapters.ClipperIntsInput
.. autoclass:: bentoml.adapters.ClipperDoublesInput
.. autoclass:: bentoml.adapters.ClipperStringsInput


API OutputAdapters
==================

BaseOutputAdapter
+++++++++++++++++
.. autoclass:: bentoml.adapters.BaseOutputAdapter

DefaultOutput
+++++++++++++
The default output adapter is used when there's no `output` specified in an Inference
API. It ensembles multiple common output adapter and dynamically choose one base on the
return value of the user-defined inference API callback function.

.. autoclass:: bentoml.adapters.DefaultOutput

DataframeOutput
+++++++++++++++
.. autoclass:: bentoml.adapters.DataframeOutput

TfTensorOutput
++++++++++++++
.. autoclass:: bentoml.adapters.TfTensorOutput

JsonSerializableOutput
++++++++++++++++++++++
.. autoclass:: bentoml.adapters.JsonSerializableOutput


Adapter Roadmap
===============

The following adapter types are on our roadmap:

* NpNdarrayInputAdapter
* NpNdarrayOutputAdapter
* FileOutputAdapter
* ImageOutputAdapter
* MultiFileOutputAdapter