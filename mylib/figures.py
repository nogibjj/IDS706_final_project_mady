import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import os

# Load the saved model
model = Word2Vec.load("data/skills_word2vec.model")

# Load the Parquet file
jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")

# Define the set of skills of interest
skills_of_interest = ["python", "sql", "machine", "learning", "analysis"]


# Function to generate a line plot for the demand for the set of skill over time
def plot_state_demand(
    skills_of_interest,
    model,
    jobpostDF,
    shapefile_path="data/tl_2024_us_state.shp",
    save_path="static/fig/",
):
    """
    Generates and saves a state-level map visualizing the demand for the specified skills.

    Parameters:
    - skills_of_interest (list): A list of skills to analyze.
    - model: Trained Word2Vec model with skill embeddings.
    - jobpostDF (DataFrame): DataFrame containing job postings with 'fips' and 'matched_skills' columns.
    - shapefile_path (str): Path to the state-level shapefile.
    - save_path (str): Directory to save the output figure.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    continental_states_fips = [
        "01",
        "04",
        "05",
        "06",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "53",
        "54",
        "55",
        "56",
    ]

    # Compute vector representation for the skills of interest
    skill_vectors = [
        model.wv[skill]
        for skill in skills_of_interest
        if skill in model.wv.key_to_index
    ]
    if not skill_vectors:
        raise ValueError(
            "None of the skills in the list are in the Word2Vec model's vocabulary."
        )
    skills_of_interest_vector = np.mean(skill_vectors, axis=0)

    # Aggregate skill embeddings by region
    state_vectors = (
        jobpostDF.groupby("state_fips")["matched_skills"]
        .apply(
            lambda skills_list: (
                np.mean(
                    [
                        model.wv[skill]
                        for skills in skills_list
                        for skill in skills
                        if skill in model.wv.key_to_index
                    ],
                    axis=0,
                )
                if len(skills_list) > 0
                else np.zeros(model.vector_size)
            )
        )
        .reset_index(name="state_vector")
    )

    # # Aggregate skill embeddings by state (using first two digits of FIPS codes)
    # region_vectors["fips"] = region_vectors["fips"].astype(str).str.zfill(5)
    # region_vectors["state_fips"] = region_vectors["fips"].str[:2]  # Extract state FIPS
    # state_vectors = (
    #     region_vectors.groupby("state_fips")["region_vector"]
    #     .apply(
    #         lambda vectors: (
    #             np.mean([vector for vector in vectors if np.any(vector)], axis=0)
    #             if len(vectors) > 0
    #             else np.zeros(model.vector_size)
    #         )
    #     )
    #     .reset_index(name="state_vector")
    # )

    # Calculate cosine similarity for states
    state_vectors["similarity"] = state_vectors["state_vector"].apply(
        lambda vector: (
            cosine_similarity([skills_of_interest_vector], [vector])[0, 0]
            if np.any(vector)
            else 0
        )
    )

    # Normalize similarity scores
    scaler = MinMaxScaler()
    state_vectors["similarity_normalized"] = scaler.fit_transform(
        state_vectors[["similarity"]]
    )

    # Load state-level shapefile
    usa_shapefile = gpd.read_file(shapefile_path)
    usa_state_shapefile = usa_shapefile[
        usa_shapefile["STATEFP"].isin(continental_states_fips)
    ]

    # Merge with geographic data
    usa_state_map = usa_state_shapefile.merge(
        state_vectors, left_on="STATEFP", right_on="state_fips", how="left"
    )
    usa_state_map["similarity_normalized"] = usa_state_map[
        "similarity_normalized"
    ].fillna(0)

    # Visualize similarity on a state-level map
    fig, ax = plt.subplots(figsize=(12, 8))
    usa_state_map.plot(
        ax=ax,
        column="similarity_normalized",
        cmap="Blues",
        legend=True,
        legend_kwds={
            "label": "Level of demand for your skills",
            "orientation": "horizontal",
        },
    )
    ax.set_title("Regional Demand for your Skills", fontsize=16, weight="bold")
    ax.set_axis_off()
    ax.set_facecolor("lightgray")

    # Save the figure
    output_file = os.path.join(save_path, "state_demand.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Map saved to {output_file}")


# Function to calculate similarity between job post skills and skills of interest
def calculate_similarity(job_skills, model, skills_of_interest_vector):
    skill_vectors = [
        model.wv[skill] for skill in job_skills if skill in model.wv.key_to_index
    ]
    if skill_vectors:
        job_skills_vector = np.mean(skill_vectors, axis=0)
        similarity = cosine_similarity(
            [skills_of_interest_vector], [job_skills_vector]
        )[0, 0]
        return similarity
    else:
        return 0


# Function to generate a line plot for the demand for the set of skill over time
def plot_skill_demand_over_time(
    jobpostDF, skills_of_interest, model, save_path="static/fig/skill_demand_plot.png"
):
    """
    Generates a line chart showing the evolution of skill demand over time and saves it to a file.

    Parameters:
    - jobpostDF (pd.DataFrame): DataFrame with columns 'converted_time' and 'matched_skills'.
    - skills_of_interest (list): List of skills of interest to track.
    - model (Word2Vec model): Pre-trained word2vec model containing word vectors.
    - save_path (str): Path where the plot should be saved (default is 'fig/skill_demand_plot.png').

    Returns:
    - None (saves the plot to a file).
    """

    # Ensure the fig directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get vector representation for the set of skills of interest
    skills_of_interest_vector = np.mean(
        [
            model.wv[skill]
            for skill in skills_of_interest
            if skill in model.wv.key_to_index
        ],
        axis=0,
    )

    # Apply the function to calculate similarity for each job post
    jobpostDF["similarity"] = jobpostDF["matched_skills"].apply(
        lambda skills: calculate_similarity(skills, model, skills_of_interest_vector)
    )

    # Convert the 'converted_time' to a period for aggregation (e.g., by month or year)
    jobpostDF["time_period"] = jobpostDF["converted_time"].dt.to_period(
        "M"
    )  # Change 'M' to 'D' for daily or 'Y' for yearly

    # Group by time period and calculate the average similarity score for each period
    time_similarity = (
        jobpostDF.groupby("time_period")["similarity"].mean().reset_index()
    )

    # Format the time_period for better x-axis labels (e.g., 'yy.mmm')
    time_similarity["time_period"] = time_similarity["time_period"].dt.strftime("%y.%b")

    # Plot the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(
        time_similarity["time_period"].astype(str),
        time_similarity["similarity"],
        marker="o",
        linestyle="-",
        color="b",
    )

    # Customize the chart
    plt.title("Evolution of skill demand per month", fontsize=14)
    plt.ylabel("Average demand", fontsize=12)
    plt.xlabel("Time (yy.mmm)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)

    # Save the plot to the specified file path
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Time plot saved to {save_path}")


# Function to calculate the average salary for a specific skill
def calculate_average_salary_for_skill(skill, jobpostDF, threshold=1000000):
    jobpostDF_filtered = jobpostDF[jobpostDF["normalized_salary"] < threshold]
    # Filter job posts that contain the skill
    filtered_jobposts = jobpostDF_filtered[
        jobpostDF_filtered["matched_skills"].apply(lambda skills: skill in skills)
    ]
    # Calculate the average salary for those job posts
    avg_salary = (
        filtered_jobposts["normalized_salary"].mean()
        if not filtered_jobposts.empty
        else 0
    )
    return avg_salary


# Function to generate a waterfall plot for the cumulative salary contribution of each skill
def plot_skill_income_waterfall(
    skills_of_interest, jobpostDF, save_path="static/fig/skill_income_waterfall.png"
):
    # Calculate average salary for each skill of interest
    avg_salaries = [
        calculate_average_salary_for_skill(skill, jobpostDF)
        for skill in skills_of_interest
    ]

    # Divide each average salary by the number of skills to get the additional income per skill
    additional_income = [
        avg_salary / len(skills_of_interest) for avg_salary in avg_salaries
    ]

    # Prepare data for the waterfall chart
    x_labels = skills_of_interest
    values = additional_income
    cumulative_values = np.cumsum(values)

    # Create the waterfall chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [label.capitalize() for label in x_labels]
    for i in range(len(values)):
        # Positive income -> Green bars; Negative income -> Red bars
        color = "green" if values[i] >= 0 else "red"
        ax.bar(
            x=i,
            height=values[i],
            bottom=cumulative_values[i] - values[i],
            color=color,
            edgecolor="black",
            label=x_labels[i - 1] if i > 0 and i < len(x_labels) + 1 else None,
        )

    # Add the final total stacked bar
    ax.bar(
        x=len(values),
        height=cumulative_values[-1],
        bottom=0,
        color="blue",
        edgecolor="black",
        label="Total",
    )

    # Customize the chart
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(x_labels, rotation=0, ha="right")
    ax.set_title(
        "Cumulative Expected Yearly Income From Your Skills in 2024", fontsize=14
    )
    ax.set_ylabel("Cumulative Expected Yearly Income", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a legend
    ax.legend(title="Skills", loc="upper left", frameon=True)

    # Save the plot to a file
    fig.savefig(
        save_path, bbox_inches="tight"
    )  # Save with tight layout to avoid clipping
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Generate the state demand map
    plot_state_demand(skills_of_interest, model, jobpostDF)

    # Generate the skill demand over time plot
    plot_skill_demand_over_time(jobpostDF, skills_of_interest, model)

    # Generate the skill income waterfall plot
    plot_skill_income_waterfall(skills_of_interest, jobpostDF)
