import xgboost
from sklearn.datasets import load_svmlight_file
from locust import HttpUser, task, between

test_data = load_svmlight_file('data/agaricus.txt.train')
num_of_rows = len(test_data[0])


class AgaricusLoadTestUser(HttpUser):

    @task
    def classify(self):
        input_data = test_data[1][np.random.choice(num_of_rows)]
        self.client.post("/classify", json=list(input_data))

    wait_time = between(0.01, 2)
