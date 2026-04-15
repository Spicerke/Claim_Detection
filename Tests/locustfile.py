from locust import HttpUser, task, between
import random
import string

class ClaimDetectionUser(HttpUser):
    # Simulates a user waiting between 1 to 3 seconds before their next action
    wait_time = between(1, 3)

    @task(1)
    def check_health(self):
        """Standard Load Test: Hit the health endpoint."""
        self.client.get("/health")

    @task(3)
    def standard_prediction(self):
        """Standard Load Test: Normal API usage."""
        unique_text = f"The Eiffel Tower is located in Paris, France. {random.randint(1, 1000000)}"
        payload = {"text": unique_text}
        self.client.post("/predict", json=payload)

    @task(1)
    def security_test_oversized_payload(self):
        """
        Security/Robustness Test: Attempt to bypass the Pydantic max_length=1000 validation.
        We expect the API to return a 422 Unprocessable Entity, NOT crash (500).
        """
        massive_string = "".join(random.choices(string.ascii_letters, k=5000))
        payload = {"text": massive_string}
        
        # We catch the response so Locust doesn't mark the expected 422 error as a "failure"
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422 for oversized payload, got {response.status_code}")

    @task(1)
    def security_test_empty_payload(self):
        """
        Security/Robustness Test: Send empty or malformed data.
        Again, we expect a 422, not a server crash.
        """
        payload = {"text": ""}
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422 for empty payload, got {response.status_code}")

    @task(1)
    def security_test_rate_limit_spam(self):
        """
        Security Test: Rapid-fire requests to test rate limiting or DDoS resilience.
        It sends 10 requests instantly without waiting.
        """
        for _ in range(10):
            payload = {"text": "Spam request"}
            # If you have a rate limiter, you'd check for a 429 Too Many Requests status here
            self.client.post("/predict", json=payload)