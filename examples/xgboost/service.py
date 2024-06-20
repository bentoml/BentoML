import os

import numpy as np
import xgboost as xgb

import bentoml


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CancerClassifier:
    # Retrieve the latest version of the model from the BentoML model store
    bento_model = bentoml.models.get("cancer:latest")

    def __init__(self):
        self.model = bentoml.xgboost.load_model(self.bento_model)

        # Check resource availability
        if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
            self.model.set_param({"predictor": "gpu_predictor", "gpu_id": 0})  # type: ignore (incomplete XGBoost types)
        else:
            nthreads = os.getenv("OMP_NUM_THREADS")
            if nthreads:
                nthreads = max(int(nthreads), 1)
            else:
                nthreads = 1
            self.model.set_param({"predictor": "cpu_predictor", "nthread": nthreads})

    @bentoml.api
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(xgb.DMatrix(data))
