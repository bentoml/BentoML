import numpy as np
import mlflow
import bentoml
from river import ensemble
from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from river import datasets
from sklearn.metrics import mean_squared_error

# Custom Python model class to include river model in mlflow
import mlflow.pyfunc

class ARFModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.arf_model = ensemble.AdaptiveRandomForestRegressor(seed=42)

    def learn_one(self,input_data,target):
        return self.arf_model.learn_one(input_data,target)
        
    def predict_one(self,model_input):
        return self.arf_model.predict_one(model_input)
        
    def predict(self,context,model_input):
        return self.predict_one(model_input)

def prepare_data():
    dataset = datasets.TrumpApproval()
    # Prepare train and test dataset
    train = list()
    test = list()
    for data in dataset.take(100):
        if len(train) < 80:
            train.append(data)
        else:
            test.append(data)
    return train,test

# train model
def train_model(model,train_data):
    for data in train_data:
        # data[0]: input feature, data[1]: label
        model.learn_one(data[0],data[1])
    return model

# evaluate model
def test_model(model,test_data):
    y_pred = list()
    y_test = list()
    for data in test_data:
        y_pred.append(model.predict_one(data[0]))
        y_test.append(data[1])
    return mean_squared_error(y_test, y_pred)

def main():
    # Initialize ARF model
    arf_model = ARFModel()
    # Load data
    train_data, test_data = prepare_data()
    # train and evaluate model
    trained_arf_model = train_model(arf_model,train_data)
    mse = test_model(trained_arf_model,test_data)
    print(f"Mean Squared Error: {mse}")

    # log model in mlflow
    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        # logging trained model 
        model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=trained_arf_model)
        # import loagged model in Bentoml model store
        bento_model = bentoml.mlflow.import_model('river_arf_model', model_info.model_uri)
        print(f"Model imported to BentoML: {bento_model}")
    mlflow.end_run()

if __name__ == "__main__":
    main()
        