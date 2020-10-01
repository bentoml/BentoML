Base Artifact
=============

All BentoML artifacts are inherited from the BentoServiceArtifact base class:

.. autoclass:: bentoml.service.artifacts.BentoServiceArtifact


In a BentoService#artifacts returns an ArtifactCollection instance:

.. autoclass:: bentoml.service.artifacts.ArtifactCollection


Common Artifacts
++++++++++++++++

PickleArtifact

.. autoclass:: bentoml.service.artifacts.common.PickleArtifact

JSONArtifact

.. autoclass:: bentoml.service.artifacts.common.JSONArtifact

TextFileArtifact

.. autoclass:: bentoml.service.artifacts.common.TextFileArtifact


.. spelling::

    deserialization
    deserializing
    stdlib
