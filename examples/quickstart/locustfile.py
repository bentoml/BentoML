import numpy as np
from locust import task
from locust import between
from locust import HttpUser
from sklearn import datasets

test_data = datasets.load_iris().data
num_of_rows = test_data.shape[0]
max_batch_size = 10


class IrisLoadTestUser(HttpUser):
    """
    Usage:
        Run the iris_classifier service in production mode:

            bentoml serve iris_classifier:latest --production

        Start locust load testing client with:

            locust -H http://localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        start = np.random.choice(num_of_rows - max_batch_size)
        end = start + np.random.choice(max_batch_size) + 1

        input_data = test_data[start:end]
        self.client.post("/classify", json=input_data.tolist())

    wait_time = between(0.01, 2)
