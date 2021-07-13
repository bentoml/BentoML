# pylint: disable=import-error
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'nested_zipmodule.zip'))
from nested_zipmodule.identity import identity
from nested_zipmodule.boolean_module.return_true import return_true
from return_false import return_false


import bentoml
from bentoml.adapters import DataframeInput
from bentoml.sklearn import SklearnModelArtifact


@bentoml.env(zipimport_archives=['nested_zipmodule.zip'])
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        df = identity(df)
        assert return_true() is True
        assert return_false() is False

        return self.artifacts.model.predict(df)
