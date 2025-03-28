from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def hello_world(self):
        out = self.client.get("/?input_text=请介绍一下沈阳")
        print(out.text)
