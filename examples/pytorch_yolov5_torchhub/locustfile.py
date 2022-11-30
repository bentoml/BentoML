from locust import task
from locust import between
from locust import HttpUser

with open("yolov5/data/images/bus.jpg", "rb") as f:
    test_image_bytes = f.read()


class PyTorchMNISTLoadTestUser(HttpUser):

    wait_time = between(0.3, 1.7)

    @task
    def predict_image(self):
        files = {"upload_files": ("bus.jpg", test_image_bytes, "image/png")}
        self.client.post("/predict_image", files=files)
