import bentoml
import xgboost as xgb

# Load the model by setting the model tag
booster = bentoml.xgboost.load_model("cancer:latest")

# Predict using a sample
res = booster.predict(xgb.DMatrix([[1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
    4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
    1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
    1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
    1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]]))
  
print(res)
# Expected output: [[0.02664177 0.9733583 ]]