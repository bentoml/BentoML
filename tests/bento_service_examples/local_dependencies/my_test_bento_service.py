import bentoml
from bentoml.handlers import DataframeHandler
from bentoml.artifact import SklearnModelArtifact

from tests.bento_service_examples.local_dependencies.my_test_dependency import (
    dummy_util_func,
)


@bentoml.env(pip_dependencies=["scikit-learn"])
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):
    @bentoml.api(DataframeHandler)
    def predict(self, df):
        df = dummy_util_func(df)

        from tests.bento_service_examples.local_dependencies.dynamically_imported_dependency import (
            dummy_util_func_ii,
        )

        df = dummy_util_func_ii(df)

        return self.artifacts.model.predict(df)
