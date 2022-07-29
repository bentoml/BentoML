from locust import HttpUser, task, between

with open("samples/1.png", "rb") as f:
    test_image_bytes = f.read()


class PyTorchMNISTLoadTestUser(HttpUser):

    wait_time = between(0.01, 2)

    @task
    def predict_image(self):
        files = {"upload_files": ("1.png", test_image_bytes, "image/png")}
        self.client.post("/predict_image", files=files)

