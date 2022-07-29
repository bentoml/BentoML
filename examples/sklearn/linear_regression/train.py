from sklearn import linear_model

import bentoml

if __name__ == "__main__":
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

    print("coef: ", reg.coef_)
    bento_model = bentoml.sklearn.save_model("linear_reg", reg)
    print(f"Model saved: {bento_model}")

    # Test running inference with BentoML runner
    test_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()
    test_runner.init_local()
    assert test_runner.predict.run([[1, 1]]) == reg.predict([[1, 1]])
