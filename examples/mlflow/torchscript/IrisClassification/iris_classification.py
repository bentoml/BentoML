# pylint: disable=abstract-method,redefined-outer-name
import argparse

import numpy as np
import torch
import torch.nn as nn
import mlflow.pytorch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import bentoml


class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data():
    iris = load_iris()
    data = iris.data
    labels = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
    )

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    return X_train, X_test, y_train, y_test, target_names


def train_model(model, epochs, X_train, y_train):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        out = model(X_train)
        loss = criterion(out, y_train).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("number of epoch", epoch, "loss", float(loss))

    return model


def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predict_out = model(X_test)
        _, predict_y = torch.max(predict_out, 1)

        print(
            "\nprediction accuracy",
            float(accuracy_score(y_test.cpu(), predict_y.cpu())),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iris Classification Torchscripted model"
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to run (default: 100)"
    )

    args = parser.parse_args()

    model = IrisClassifier()
    model = model.to(device)
    X_train, X_test, y_train, y_test, target_names = prepare_data()
    scripted_model = torch.jit.script(model)  # scripting the model
    scripted_model = train_model(scripted_model, args.epochs, X_train, y_train)
    test_model(scripted_model, X_test, y_test)

    # Saving model and running inference with BentoML:

    # Option1: save natively with bentoml.torchscript_iris
    bentoml.torchscript.save_model(
        "torchscript_iris", scripted_model, signatures={"__call__": {"batchable": True}}
    )
    model_runner = bentoml.torchscript.get("torchscript_iris").to_runner()
    model_runner.init_local()

    test_input = np.array([4.4000, 3.0000, 1.3000, 0.2000], dtype="float32")
    actual = "setosa"
    prediction = model_runner.run(test_input)
    predicted = target_names[np.argmax(prediction)]
    print("\nPREDICTION RESULT: ACTUAL: {}, PREDICTED: {}".format(actual, predicted))

    # Option2: save MLflow model and import MLflow pyfunc model to BentoML
    with mlflow.start_run() as run:
        # logging scripted model
        mlflow.pytorch.log_model(scripted_model, "model")

        # Import logged mlflow model to BentoML model store for serving:
        model_uri = mlflow.get_artifact_uri("model")
        bento_model = bentoml.mlflow.import_model(
            "mlflow_torch_iris", model_uri, signatures={"predict": {"batchable": True}}
        )
        print(f"Model imported to BentoML: {bento_model}")

        model_runner = bentoml.mlflow.get("mlflow_torch_iris").to_runner()
        model_runner.init_local()

        test_input = np.array([4.4000, 3.0000, 1.3000, 0.2000], dtype="float32")
        actual = "setosa"
        prediction = model_runner.predict.run(test_input)
        predicted = target_names[np.argmax(prediction)]
        print(
            "\nPREDICTION RESULT: ACTUAL: {}, PREDICTED: {}".format(actual, predicted)
        )
