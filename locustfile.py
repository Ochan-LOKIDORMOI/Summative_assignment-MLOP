from locust import HttpUser, task, between
import os


class AnimalClassificationLoadTest(HttpUser):
    """
    Simulates user behavior for testing the Animal Classification APIs.
    """
    wait_time = between(1, 2)  # Simulate users waiting 1 to 3 seconds between requests

    # Dynamically determine paths for test files
    base_path = os.path.dirname(__file__)  # Get the directory of the locustfile
    image_path = os.path.join(base_path, "locust_test_image", "test_image.jpg")
    zip_path = os.path.join(base_path, "locust_test_image", "dataimages.zip")

    def on_start(self):
        """
        Runs once per simulated user when the load test starts.
        Checks for the existence of required test files.
        """
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Test image not found at {self.image_path}")

        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Test zip file not found at {self.zip_path}")

    @task
    def test_fastapi_predict(self):
        """
        Test the `/predict` endpoint of the FastAPI application.
        """
        with open(self.image_path, "rb") as image:
            files = {"file": ("test_image.jpg", image, "image/jpeg")}
            response = self.client.post("/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result.get('animal_type')} for file: {result.get('filename')}")
        else:
            print(f"FastAPI Predict failed with status: {response.status_code}")

    @task
    def test_fastapi_health_check(self):
        """
        Test the root endpoint (`/`) of the FastAPI application.
        """
        response = self.client.get("/")
        if response.status_code == 200:
            print("FastAPI Health Check Passed!")
        else:
            print("FastAPI Health Check Failed!")

    @task
    def test_flask_index(self):
        """
        Test the Flask `/` route to ensure the UI loads correctly.
        """
        response = self.client.get("/")
        if response.status_code == 200:
            print("Flask Index Loaded Successfully!")
        else:
            print("Flask Index Failed!")

    @task
    def test_flask_retrain(self):
        """
        Test the Flask `/upload` route with a dummy zip file.
        """
        if os.path.exists(self.zip_path):
            with open(self.zip_path, "rb") as zip_file:
                files = {"file": ("dataimages.zip", zip_file, "application/zip")}
                response = self.client.post("/upload", files=files)

            if response.status_code == 200:
                print("Flask Retrain Route Worked Successfully!")
            else:
                print(f"Flask Retrain Failed: {response.status_code}")
        else:
            print(f"Zip file for retraining not found at {self.zip_path}")
