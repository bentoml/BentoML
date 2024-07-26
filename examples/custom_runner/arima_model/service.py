import numpy as np
import typing
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

arima_model = bentoml.picklable_model.get("arima_forecast_model:latest")

# Custom runner for ARIMA model
class ARIMAForecastRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        super().__init__()
        self.model = arima_model
    
    @bentoml.Runnable.method(batchable=False)
    def forecast(self, values_to_forecast):
        # Load trained arima model from bentoml
        arima_model = self.model.load_model()
        model_fit = arima_model.fit()
        predictions = model_fit.forecast(int(values_to_forecast))
        return predictions

arima_forecast_runner = bentoml.Runner(ARIMAForecastRunnable)

svc = bentoml.Service("arima_model_forecast", runners=[arima_forecast_runner], models=[arima_model])

@svc.api(input=NumpyNdarray(dtype="int"), output=JSON())
def predict_forecast(input_data: np.ndarray):
    return arima_forecast_runner.forecast.run(input_data[0])

