# Deprecated

The `bentoml.artifact` module is now deprecated.

Use`bentoml.frameworks.*` and `bentoml.service.*` instead.

e.g.:
```
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.common_artifacts import PickleArtifact
from bentoml.service.artifacts import BentoServiceArtifact
```
