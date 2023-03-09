import numpy as np
import pandas as pd
from sample import sample_input

import bentoml
from bentoml.io import JSON
from bentoml.io import PandasDataFrame

model_ref = bentoml.xgboost.get("ieee-fraud-detection-lg:latest")
preprocessor = model_ref.custom_objects["preprocessor"]
fraud_model_runner = model_ref.to_runner()
fraud_model_runner.init_local()

svc = bentoml.Service("fraud_detection")

input_spec = PandasDataFrame.from_sample(sample_input)


@svc.api(input=input_spec, output=JSON())
def is_fraud(input_df: pd.DataFrame):
    input_df = input_df.astype(sample_input.dtypes)
    input_features = preprocessor.transform(input_df)
    results = fraud_model_runner.predict_proba.run(input_features)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {"is_fraud": list(map(bool, predictions)), "is_fraud_prob": results[:, 1]}


@svc.api(input=input_spec, output=JSON())
async def is_fraud_async(input_df: pd.DataFrame):
    input_df = input_df.astype(sample_input.dtypes)
    input_features = preprocessor.transform(input_df)
    results = await fraud_model_runner.predict_proba.async_run(input_features)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {"is_fraud": list(map(bool, predictions)), "is_fraud_prob": results[:, 1]}
