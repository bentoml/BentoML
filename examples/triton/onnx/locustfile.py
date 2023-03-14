from __future__ import annotations

from locust import task
from locust import constant
from locust import HttpUser


class TritonOnnxYolov5User(HttpUser):
    wait_time = constant(0)
    filename = "./data/zidane.jpg"

    @task
    def triton_onnx_yolov5_infer(self):
        self.client.post(
            "/triton_onnx_yolov5_infer",
            files=[("file", (self.filename, open(self.filename, "rb"), "image/jpeg"))],
        )


class BentoOnnxYolov5User(HttpUser):
    wait_time = constant(0)
    filename = "./data/zidane.jpg"

    @task
    def bentoml_onnx_yolov5_infer(self):
        self.client.post(
            "/bentoml_onnx_yolov5_infer",
            files=[("file", (self.filename, open(self.filename, "rb"), "image/jpeg"))],
        )
