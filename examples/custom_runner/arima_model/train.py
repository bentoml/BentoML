# import libraries
import numpy as np
import pandas as pd
import bentoml
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def main():
        # Load the dataset
        data = sm.datasets.sunspots.load_pandas().data
        # Prepare dataset
        data.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
        data.index.freq = data.index.inferred_freq
        del data["YEAR"]

        # Split into train and test sets
        X = data.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        # Create a list of records to train ARIMA
        history = [x for x in train]
        # Create a list to store the predicted values
        predictions = list()

        # Iterate over the test data
        for t in range(len(test)):
            model = ARIMA(history, order=(5, 1, 0))
            # fit the model and create forecast
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        
        y_test = test
        y_pred = predictions

        # Calculate root mean squared error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("Root Mean Squared Error:", rmse)
        
        # Save model with BentoML
        saved_model = bentoml.picklable_model.save_model(
                        "arima_forecast_model",
                        model,
                        custom_objects= {"historical_data": history,},
                                        signatures={"predict": {"batchable": True}},
                )

        print(f"Model saved: {saved_model}")

if __name__ == "__main__":
    main()