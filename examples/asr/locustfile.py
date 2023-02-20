from locust import task
from locust import constant
from locust import HttpUser


class TranscribeUser(HttpUser):
    wait_time = constant(0)

    @task
    def transcribe(self):
        self.client.post("/transcribe", data="./samples/jfk.wav")
