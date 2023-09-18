from __future__ import annotations

from locust import task
from locust import constant
from locust import HttpUser


class TritonTorchscriptYolov5User(HttpUser):
    wait_time = constant(0)
    filename = "./data/zidane.jpg"

    @task
    def triton_torchscript_yolov5_infer(self):
        self.client.post(
            "/triton_torchscript_yolov5_infer",
            files=[("file", (self.filename, open(self.filename, "rb"), "image/png"))],
        )


class BentoTorchscriptYolov5User(HttpUser):
    wait_time = constant(0)
    filename = "./data/zidane.jpg"

    @task
    def bentoml_torchscript_yolov5_infer(self):
        self.client.post(
            "/bentoml_torchscript_yolov5_infer",
            files=[("file", (self.filename, open(self.filename, "rb"), "image/jpeg"))],
        )
