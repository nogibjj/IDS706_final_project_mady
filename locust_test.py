from locust import HttpUser, between, task
class WebsiteUser(HttpUser):
    """
    Simulated user for testing the front page of the website.
    """
    host = "https://jdwhnfkggk.us-west-2.awsapprunner.com/"  # Replace with your website's URL if needed
    wait_time = between(2, 5)  # Wait between 2 to 5 seconds between requests
    @task
    def front_page(self):
        self.client.get("/")  # Test only the front page


