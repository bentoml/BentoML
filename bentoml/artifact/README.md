# Deprecated

The `bentoml.artifact` module is now deprecated.

Use`bentoml.frameworks.*` and `bentoml.artifact.common.*` instead.

e.g.:
```
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.artifact.common import PickleArtifact
from bentoml.service.artifacts import BentoServiceArtifact
```
