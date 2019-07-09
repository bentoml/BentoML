API Reference
=============

BentoML
*******

BentoService
++++++++++++
.. autoclass:: bentoml.service.BentoService

  .. automethod:: bentoml.service.BentoService.name

  .. automethod:: bentoml.service.BentoService.version

  .. automethod:: bentoml.service.BentoService.get_service_apis


.. autoclass:: bentoml.service.BentoServiceAPI

api
+++
.. autofunction:: bentoml.api

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

.. _api-load-ref:

load
++++
.. autofunction:: bentoml.load

config
++++++
.. autofunction:: bentoml.config


Handlers
********

DataframeHandler
++++++++++++++++
.. autoclass:: bentoml.handlers.DataframeHandler

ImageHandler
++++++++++++
.. autoclass:: bentoml.handlers.ImageHandler

FastaiImageHandler
++++++++++++++++++
.. autoclass:: bentoml.handlers.FastaiImageHandler

JsonHandler
+++++++++++
.. autoclass:: bentoml.handlers.JsonHandler


Artifacts
*********

PickleArtifact
++++++++++++++
.. autoclass:: bentoml.artifact.PickleArtifact

TextFileArtifact
++++++++++++++++
.. autoclass:: bentoml.artifact.TextFileArtifact

PytorchModelArtifact
++++++++++++++++++++
.. autoclass:: bentoml.artifact.PytorchModelArtifact

XgboostModelArtifact
++++++++++++++++++++
.. autoclass:: bentoml.artifact.XgboostModelArtifact

FastaiModelArtifact
+++++++++++++++++++
.. autoclass:: bentoml.artifact.FastaiModelArtifact

H2oModelArtifact
++++++++++++++++
.. autoclass:: bentoml.artifact.H2oModelArtifact

TfKerasModelArtifact
++++++++++++++++++++
.. autoclass:: bentoml.artifact.TfKerasModelArtifact


Exceptions
**********

BentoMLException
++++++++++++++++
.. autoexception:: bentoml.exceptions.BentoMLException
