import numpy as np

import bentoml


@bentoml.service(
    resources={
        "cpu": "2",
        "memory": "2Gi",
    },
)
class ArimaForecast:
    """
    A simple ARIMA model to forecast future predictions
    """

    # Load in the class scope to declare the model as a dependency of the service
    arima_model = bentoml.picklable_model.get("arima_forecast_model:latest")

    def __init__(self):
        """
        Initialize the service by loading the model from the model store
        """
        self.model = self.arima_model.load_model()

    @bentoml.api
    def forecast(self, data: np.ndarray) -> np.ndarray:
        """
        Define API with input as number of forecasts to predict in the future
        """
        model_fit = self.model.fit()
        predictions = model_fit.forecast(int(data))
        return predictions
