import bentoml
import pandas as pd
import numpy as np
from bentoml.io import NumpyNdarray, PandasDataFrame, PandasSeries, JSON
import bentoml.sklearn

input_spec = PandasDataFrame.from_sample(pd.DataFrame(np.array([[5,4,3,2]])))

runner = bentoml.sklearn.load_runner("sklearn_model_clf")

svc = bentoml.Service("server", runners=[runner])

@svc.api(input=input_spec, output=JSON(orient='records'))
def predict(input_arr):
    res = runner.run_batch(input_arr)
    return {"res":pd.DataFrame(res)}

app = svc.asgi_app
