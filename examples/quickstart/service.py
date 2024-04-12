import numpy as np
from typing_extensions import Annotated

import bentoml
from bentoml.validators import Shape

from pydantic import Field

@bentoml.service(
    resources={
        "cpu": "1",
        "memory": "2Gi",
    },
)
class IrisClassifier:
    '''
    A simple Iris classification service using a sklearn model
    '''

    # Load in the class scope to declare the model as a dependency of the service
    iris_model = bentoml.models.get("iris_sklearn:latest")

    def __init__(self):
        '''
        Initialize the service by loading the model from the model store
        '''
        import joblib

        self.model = joblib.load(self.iris_model.path_of("model.pkl"))

    @bentoml.api
    def classify(
        self,
        input_series: Annotated[np.ndarray, Shape((-1, 4))] = Field(default=[[5.2, 2.3, 5.0, 0.7]]),
    ) -> np.ndarray:
        '''
        Define API with preprocessing and model inference logic
        '''
        return self.model.predict(input_series)
