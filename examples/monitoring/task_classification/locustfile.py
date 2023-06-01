import numpy as np
from locust import task
from locust import between
from locust import HttpUser
from sklearn import datasets

test_data = datasets.load_iris().data
num_of_rows = test_data.shape[0]


class IrisHttpUser(HttpUser):
    """
    Usage:
        Run the iris_classifier service in production mode:

            bentoml serve-http iris_classifier:latest --production

        Start locust load testing client with:

            locust --class-picker -H http://localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        index = np.random.choice(num_of_rows - 1)

        input_data = test_data[index]
        self.client.post("/classify", json=input_data.tolist())

    wait_time = between(0.01, 2)
