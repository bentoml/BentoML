import sys
from os import path

import bentoml
from bentoml.adapters import DataframeInput
from bentoml.sklearn import SklearnModelArtifact

# noqa # pylint: disable=import-error
sys.path.append(path.dirname(path.abspath(__file__)))  # isort:skip
from local_dependencies.local_module import (  # isort:skip
    dependency_in_local_module_directory,
)
from local_dependencies.my_test_dependency import dummy_util_func  # isort:skip
from local_dependencies.nested_dependency import nested_dependency_func  # isort:skip


@bentoml.env(pip_packages=["scikit-learn"])
@bentoml.artifacts([SklearnModelArtifact("model")])
class IrisClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        df = dummy_util_func(df)
        df = dependency_in_local_module_directory(df)
        df = nested_dependency_func(df)

        from local_dependencies.dynamically_imported_dependency import (  # noqa: E501
            dynamically_imported_dependency_func,
        )

        df = dynamically_imported_dependency_func(df)

        return self.artifacts.model.predict(df)
