import numpy as np
import pandas as pd
from locust import task
from locust import between
from locust import HttpUser

NUM_OF_ROWS = 500
test_transactions = pd.read_csv("../data/test_transaction.csv")[0:NUM_OF_ROWS]

endpoint = "/is_fraud"
# endpoint = "/is_fraud_async"


class FraudDetectionUser(HttpUser):
    @task
    def is_fraud(self):
        index = np.random.choice(NUM_OF_ROWS)
        input_data = test_transactions[index : index + 1]
        self.client.post(endpoint, data=input_data.to_json())

    wait_time = between(0.01, 2)
