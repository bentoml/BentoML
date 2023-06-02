from locust import task
from locust import between
from locust import HttpUser

with open("samples/0.png", "rb") as f:
    test_image_bytes = f.read()


class TensorFlow2MNISTLoadTestUser(HttpUser):
    wait_time = between(0.9, 1.1)

    @task
    def predict_image(self):
        files = {"upload_files": ("1.png", test_image_bytes, "image/png")}
        self.client.post("/predict_image", files=files)
