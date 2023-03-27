import asyncio

import numpy as np
import pandas as pd
from sample import sample_input

import bentoml
from bentoml.io import JSON
from bentoml.io import PandasDataFrame

fraud_detection_preprocessors = []
fraud_detection_runners = []

for model_name in [
    "ieee-fraud-detection-0",
    "ieee-fraud-detection-1",
    "ieee-fraud-detection-2",
]:
    model_ref = bentoml.xgboost.get(model_name)
    fraud_detection_preprocessors.append(model_ref.custom_objects["preprocessor"])
    fraud_detection_runners.append(model_ref.to_runner())

svc = bentoml.Service("fraud_detection", runners=fraud_detection_runners)


@svc.api(input=PandasDataFrame.from_sample(sample_input), output=JSON())
async def is_fraud(input_df: pd.DataFrame):
    input_df = input_df.astype(sample_input.dtypes)

    async def _is_fraud(preprocessor, runner, input_df):
        input_features = preprocessor.transform(input_df)
        results = await runner.predict_proba.async_run(input_features)
        predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
        return bool(predictions[0])

    # Simultaeously run all models
    results = await asyncio.gather(
        *[
            _is_fraud(p, r, input_df)
            for p, r in zip(fraud_detection_preprocessors, fraud_detection_runners)
        ]
    )

    # Return fraud if at least one model returns fraud
    return any(results)
