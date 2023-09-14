import xgboost

import bentoml

if __name__ == "__main__":
    # read in data
    dtrain = xgboost.DMatrix("data/agaricus.txt.train")

    # specify parameters via dictionary
    param = {
        "booster": "dart",
        "max_depth": 2,
        "eta": 1,
        "objective": "binary:logistic",
    }
    num_round = 2
    bst = xgboost.train(param, dtrain, num_round)

    bentoml.xgboost.save_model("agaricus", bst)
