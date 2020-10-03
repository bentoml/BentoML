# Deprecated

The `bentoml.artifact` module is now deprecated.

Use`bentoml.frameworks.*` and `bentoml.service.artifacts.*` instead.

e.g.:
```
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.artifacts.common import PickleArtifact
```
