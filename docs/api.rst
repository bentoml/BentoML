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
.. autofunction:: bentoml.service.api_decorator

env
+++
.. autofunction:: bentoml.service.env_decorator

artifacts
+++++++++
.. autofunction:: bentoml.service.artifacts_decorator

ver
+++
.. autofunction:: bentoml.service.ver_decorator

save
++++
.. autofunction:: bentoml.archive.archiver.save

.. _api-load-ref:

load
++++
.. autofunction:: bentoml.archive.loader.load

Handlers
********

DataframeHandler
++++++++++++++++
.. autoclass:: bentoml.handlers.DataframeHandler

ImageHandler
++++++++++++
.. autoclass:: bentoml.handlers.ImageHandler

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
