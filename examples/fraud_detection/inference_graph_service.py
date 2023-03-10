import asyncio

import numpy as np
import pandas as pd
from sample import sample_input

import bentoml
from bentoml.io import JSON
from bentoml.io import PandasDataFrame

model_ref = bentoml.xgboost.get("ieee-fraud-detection-lg:latest")
preprocessor = model_ref.custom_objects["preprocessor"]

fraud_model_tiny_runner = bentoml.xgboost.get(
    "ieee-fraud-detection-tiny:latest"
).to_runner()
fraud_model_small_runner = bentoml.xgboost.get(
    "ieee-fraud-detection-sm:latest"
).to_runner()
fraud_model_large_runner = bentoml.xgboost.get(
    "ieee-fraud-detection-lg:latest"
).to_runner()

svc = bentoml.Service(
    "fraud_detection_inference_graph",
    runners=[
        fraud_model_tiny_runner,
        fraud_model_small_runner,
        fraud_model_large_runner,
    ],
)

input_spec = PandasDataFrame.from_sample(sample_input)


async def _is_fraud_async(
    runner: bentoml.Runner,
    input_df: pd.DataFrame,
):
    results = await runner.predict_proba.async_run(input_df)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {"is_fraud": list(map(bool, predictions)), "is_fraud_prob": results[:, 1]}


@svc.api(input=input_spec, output=JSON())
async def is_fraud(input_df: pd.DataFrame):
    input_df = input_df.astype(sample_input.dtypes)
    input_df = preprocessor.transform(input_df)
    return await asyncio.gather(
        _is_fraud_async(fraud_model_tiny_runner, input_df),
        _is_fraud_async(fraud_model_small_runner, input_df),
        _is_fraud_async(fraud_model_large_runner, input_df),
    )
