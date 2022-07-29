import numpy as np
from locust import task
from locust import between
from locust import HttpUser
from sklearn.datasets import load_svmlight_file

test_data = load_svmlight_file("data/agaricus.txt.train")
num_of_rows = len(test_data[0])


class AgaricusLoadTestUser(HttpUser):
    @task
    def classify(self):
        input_data = test_data[1][np.random.choice(num_of_rows)]
        self.client.post("/classify", json=list(input_data))

    wait_time = between(0.01, 2)
