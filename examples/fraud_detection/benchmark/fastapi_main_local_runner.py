import io

import numpy as np
import pandas as pd
from sample import sample_input
from fastapi import FastAPI
from fastapi import Request

import bentoml

model_ref = bentoml.xgboost.get("ieee-fraud-detection-lg:latest")
preprocessor = model_ref.custom_objects["preprocessor"]
fraud_model_runner = model_ref.to_runner()
fraud_model_runner.init_local()

app = FastAPI()


@app.post("/is_fraud")
async def is_fraud(request: Request):
    body = await request.body()
    input_df = pd.read_json(io.BytesIO(body), dtype=True, orient="records")
    input_df = input_df.astype(sample_input.dtypes)
    input_features = preprocessor.transform(input_df)
    results = await fraud_model_runner.predict_proba.async_run(input_features)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {
        "is_fraud": list(map(bool, predictions)),
        "is_fraud_prob": results[:, 1].tolist(),
    }
