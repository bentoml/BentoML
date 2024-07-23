import numpy as np
import typing
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from statsmodels.tsa.arima.model import ARIMA

# Custom runner for ARIMA model
class ARIMAForecastRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        super().__init__()
    
    @bentoml.Runnable.method(batchable=False)
    def forecast(self, test_data):
        predictions = []
        arima_model = bentoml.picklable_model.get("arima_forecast_model:latest")
        history_data = arima_model.custom_objects["historical_data"]

        # Define the ARIMA tuning parameters
        model = ARIMA(history_data, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        y_pred = output[0]
        predictions.append(y_pred)
        obs = test_data
        # Update history with single test value
        history_data.append(obs)
        # Save model with BentoML
        bentoml.picklable_model.save_model('arima_forecast_model',
                                            model,
                                            custom_objects= {"historical_data": history_data},
                                            signatures={"predict": {"batchable": True}},
        )
        return y_pred


arima_model = bentoml.Runner(ARIMAForecastRunnable)

svc = bentoml.Service("arima_model_forecast", runners=[arima_model])

@svc.api(input=NumpyNdarray(dtype="float"), output=JSON())
def predict(input_data: np.ndarray) -> typing.List[float]:
    return arima_model.forecast.run(input_data)