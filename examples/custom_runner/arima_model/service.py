import numpy as np
import typing
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

# Custom runner for ARIMA model
class ARIMAForecastRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        super().__init__()
        self.model = bentoml.picklable_model.get("arima_forecast_model:latest")
    
    @bentoml.Runnable.method(batchable=False)
    def forecast(self, values):
        # Load trained arima model from bentoml
        arima_model = self.model.load_model()
        model_fit = arima_model.fit()
        y_pred = model_fit.forecast(int(values))
        return y_pred


arima_forecast_runner = bentoml.Runner(ARIMAForecastRunnable)

svc = bentoml.Service("arima_model_forecast", runners=[arima_forecast_runner])

@svc.api(input=NumpyNdarray(dtype="int"), output=JSON())
def predict_forecast(input_data: np.ndarray):
    print(input_data)
    return arima_forecast_runner.forecast.run(input_data[0])

