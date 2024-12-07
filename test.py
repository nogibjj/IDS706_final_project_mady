import os
import pandas as pd
import numpy as np
from mylib.figures import (
    plot_state_demand,
    plot_skill_demand_over_time,
    plot_skill_income_waterfall,
)
from mylib.resume_summary import get_summary, get_top_skills
from gensim.models import Word2Vec


# test resume summary
def test_resume_summary():
    summary_text = get_summary("test_resume/example_resume.pdf")
    assert get_summary("test_resume/example_resume.pdf") is not None
    assert get_top_skills(summary_text) is not None


# Mock Word2Vec Model
class MockWord2Vec:
    def __init__(self, vocab):
        self.key_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.vector_size = 50
        self.wv = self

    def __getitem__(self, word):
        np.random.seed(self.key_to_index[word])  # Deterministic vectors for testing
        return np.random.rand(self.vector_size)


# Test setup
def test_plot_state_demand():
    mock_vocab = ["python", "sql", "machine", "learning", "analysis"]
    mock_model = MockWord2Vec(mock_vocab)
    mock_jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")
    shapefile_path = "data/tl_2024_us_state.shp"  # Update to your test shapefile path
    save_path = "static/fig/"
    output_file = os.path.join(save_path, "state_demand.png")

    # Ensure test directory exists
    os.makedirs(save_path, exist_ok=True)

    # Run the function
    plot_state_demand(
        skills_of_interest=["python", "sql", "analysis"],
        model=mock_model,
        jobpostDF=mock_jobpostDF,
        shapefile_path=shapefile_path,
        save_path=save_path,
    )

    # Assert file creation
    assert os.path.exists(output_file), "Output file was not created."

    # Clean up
    os.remove(output_file)
    print("Test passed! skills_demand_map.png created successfully.")


# Function to test plot_skill_demand_over_time
def test_plot_skill_demand_over_time():
    # Mock data for testing
    jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")
    model = Word2Vec.load("data/skills_word2vec.model")

    # Skills of interest
    skills_of_interest = ["python", "sql", "machine", "learning", "analysis"]

    # Run the function to test
    plot_skill_demand_over_time(
        jobpostDF,
        skills_of_interest,
        model,
        save_path="static/fig/test_skill_demand_plot.png",
    )

    # Assert that the file has been created
    assert os.path.exists(
        "static/fig/test_skill_demand_plot.png"
    ), "Test failed! Plot not saved."
    print("Test passed! Time Plot saved successfully.")

    # Clean up
    os.remove("static/fig/test_skill_demand_plot.png")


def test_plot_function():
    # Example DataFrame for testing
    jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")

    # Define path for saving the plot
    save_path = "static/fig/skill_income_waterfall_test.png"

    # Test the plot function by saving the plot
    plot_skill_income_waterfall(
        ["python", "sql", "machine", "learning", "analysis"], jobpostDF, save_path
    )

    # Assert that the file has been created
    assert os.path.exists(
        "static/fig/skill_income_waterfall_test.png"
    ), "Test failed! Plot not saved."
    print("Test passed! Skill bar Plot saved successfully.")

    # Check if the file is created
    if os.path.exists(save_path):
        print(f"Test passed: File saved successfully at {save_path}")
        # Optionally remove the file after test
        os.remove(save_path)
    else:
        print("Test failed: File was not created.")

    # Test invalid skills scenario
    invalid_save_path = "static/fig/invalid_skill_plot.png"
    plot_skill_income_waterfall(["nonexistent_skill"], jobpostDF, invalid_save_path)
    os.remove(invalid_save_path)


if __name__ == "__main__":
    test_resume_summary()
    test_plot_state_demand()
    test_plot_skill_demand_over_time()
    test_plot_function()
