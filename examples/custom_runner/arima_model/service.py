import numpy as np
import typing
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from statsmodels.tsa.arima.model import ARIMA

arima_model_runner = bentoml.picklable_model.get("arima_forecast_model:latest")
history_data = arima_model_runner.custom_objects["historical_data"]

# Custom runner for ARIMA model
class ARIMAForecastRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.arima = ARIMA
    
    @bentoml.Runnable.method(batchable=False)
    def forecast(self ,test_data):
        predictions = []
        history_data = arima_model_runner.custom_objects["historical_data"]

        # Define the ARIMA tuning parameters
        model = ARIMA(history_data, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        y_pred = output[0]
        predictions.append(y_pred)
        obs = test_data # single test value
        history_data.append(obs)
        print("New history data: ",history_data)
        # Save model with BentoML
        bentoml.picklable_model.save_model('arima_forecast_model',
                                            model,
                                            custom_objects= {"historical_data": history_data},
                                            signatures={"predict": {"batchable": True}},
        )
        return y_pred


arima_forecast_custom_runner = bentoml.Runner(ARIMAForecastRunnable)

svc = bentoml.Service("arima_model_forecast", runners=[arima_forecast_custom_runner])

@svc.api(input=NumpyNdarray(dtype="float"), output=JSON())
def predict(input_data: np.ndarray) -> typing.List[float]:
    return arima_forecast_custom_runner.forecast.run(input_data)