import bentoml
from bentoml.adapters import DataframeInput
from bentoml.sklearn import SklearnModelArtifact
from tests.bento_services.local_dependencies.local_module import (
    dependency_in_local_module_directory,
)
from tests.bento_services.local_dependencies.my_test_dependency import dummy_util_func
from tests.bento_services.local_dependencies.nested_dependency import (
    nested_dependency_func,
)


@bentoml.env(pip_packages=["scikit-learn"])
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        df = dummy_util_func(df)
        df = dependency_in_local_module_directory(df)
        df = nested_dependency_func(df)

        from tests.bento_services.local_dependencies.dynamically_imported_dependency import (  # noqa: E501
            dynamically_imported_dependency_func,
        )

        df = dynamically_imported_dependency_func(df)

        return self.artifacts.model.predict(df)
