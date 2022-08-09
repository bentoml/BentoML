from locust import HttpUser, task, between

class MnistTestUser(HttpUser):
    @task
    def predict(self):
        url = '/predict'
        filename = './mnist_png/testing/9/1000.png'
        files = [
            ('file', (filename, open(filename, 'rb'), 'image/png')),
        ]
        self.client.post(url, files=files)

    wait_time = between(0.05, 2)
