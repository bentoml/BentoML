import grpc
import numpy as np
import time

from locust import task
from locust import between
from locust import HttpUser
from locust import User
from sklearn import datasets

from bentoml.grpc.utils import import_generated_stubs

test_data = datasets.load_iris().data
num_of_rows = test_data.shape[0]
max_batch_size = 10


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
        start = np.random.choice(num_of_rows - max_batch_size)
        end = start + np.random.choice(max_batch_size) + 1

        input_data = test_data[start:end]
        self.client.post("/classify", json=input_data.tolist())

    wait_time = between(0.01, 2)


class GrpcUser(User):
    abstract = True

    stub_class = None

    def __init__(self, environment):
        super().__init__(environment)
        self.environment = environment
        self.pb, self.services = import_generated_stubs()
        self.channel = grpc.insecure_channel(self.host)
        self.stub = self.services.BentoServiceStub(self.channel)


class IrisGrpcUser(GrpcUser):
    """
    Implementation is inspired by https://docs.locust.io/en/stable/testing-other-systems.html

    Usage:
        Run the iris_classifier service in production mode:

            bentoml serve-grpc iris_classifier:latest --production

        Start locust load testing client with:

            locust --class-picker -H localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        start = np.random.choice(num_of_rows - max_batch_size)
        end = start + np.random.choice(max_batch_size) + 1
        input_data = test_data[start:end]
        request_meta = {
            "request_type": "grpc",
            "name": "classify",
            "start_time": time.time(),
            "response_length": 0,
            "exception": None,
            "context": None,
            "response": None,
        }
        start_perf_counter = time.perf_counter()
        try:
            request_meta["response"] = self.stub.Call(
                request=self.pb.Request(
                    api_name=request_meta["name"],
                    ndarray=self.pb.NDArray(
                        dtype=self.pb.NDArray.DTYPE_FLOAT,
                        # shape=(1, 4),
                        shape=(len(input_data), 4),
                        # float_values=[5.9, 3, 5.1, 1.8],
                        float_values=input_data.flatten(),
                    ),
                )
            )
        except grpc.RpcError as e:
            request_meta["exception"] = e
        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000
        self.environment.events.request.fire(**request_meta)
    
    wait_time = between(0.01, 2)
