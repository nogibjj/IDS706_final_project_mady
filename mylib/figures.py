# START OF CODE

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec


# Load the saved model
model = Word2Vec.load("data/skills_word2vec.model")

# Load the Parquet file
jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")


# Function to generate a state-level map plot
def plot_state_demand(
    skills_of_interest,
    model,
    jobpostDF,
    shapefile_path="data/tl_2024_us_state.shp",
    save_path="static/fig/state_demand_plot.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Generating state demand plot at {save_path}...")

    skill_vectors = [
        model.wv[skill]
        for skill in skills_of_interest
        if skill in model.wv.key_to_index
    ]
    if not skill_vectors:
        print("Error: None of the skills are in the Word2Vec model's vocabulary.")
        return

    skills_of_interest_vector = np.mean(skill_vectors, axis=0)

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

    state_vectors["similarity"] = state_vectors["state_vector"].apply(
        lambda vector: (
            cosine_similarity([skills_of_interest_vector], [vector])[0, 0]
            if np.any(vector)
            else 0
        )
    )

    scaler = MinMaxScaler()
    state_vectors["similarity_normalized"] = scaler.fit_transform(
        state_vectors[["similarity"]]
    )

    usa_shapefile = gpd.read_file(shapefile_path)
    usa_state_shapefile = usa_shapefile[
        usa_shapefile["STATEFP"].isin(
            [
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
        )
    ]

    usa_state_map = usa_state_shapefile.merge(
        state_vectors, left_on="STATEFP", right_on="state_fips", how="left"
    )
    usa_state_map["similarity_normalized"] = usa_state_map[
        "similarity_normalized"
    ].fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8))
    # Set background color
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#F7F8FA")
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
    # ax.set_title("Regional Demand for Your Skills", fontsize=16, weight="bold")
    ax.set_axis_off()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"State demand plot saved to {save_path}.")


# Function to generate a skill demand over time plot
def plot_skill_demand_over_time(
    jobpostDF, skills_of_interest, model, save_path="static/fig/skill_demand_plot.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Generating skill demand over time plot at {save_path}...")

    skills_of_interest_vector = np.mean(
        [
            model.wv[skill]
            for skill in skills_of_interest
            if skill in model.wv.key_to_index
        ],
        axis=0,
    )

    jobpostDF["similarity"] = jobpostDF["matched_skills"].apply(
        lambda skills: (
            cosine_similarity(
                [skills_of_interest_vector],
                np.mean(
                    [
                        model.wv[skill]
                        for skill in skills
                        if skill in model.wv.key_to_index
                    ],
                    axis=0,
                )[np.newaxis],
            )[0, 0]
            if len(skills) > 0
            else 0
        )
    )

    jobpostDF["time_period"] = jobpostDF["converted_time"].dt.to_period("M")

    time_similarity = (
        jobpostDF.groupby("time_period")["similarity"].mean().reset_index()
    )

    time_similarity["time_period"] = time_similarity["time_period"].dt.strftime("%y.%b")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set background color
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#F7F8FA")

    # Plot the data
    ax.plot(
        time_similarity["time_period"].astype(str),
        time_similarity["similarity"],
        marker="o",
        linestyle="-",
        color="b",
    )

    # Remove the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Add grid
    ax.grid(True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Skill demand over time plot saved to {save_path}.")


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


# Function to generate skill income waterfall plot
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
    # Set background color
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#F7F8FA")
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

    # Remove the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Add grid
    ax.grid(True)
    # Customize the chart
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(x_labels, rotation=0, ha="right")
    # ax.set_title(
    #    "Cumulative Expected Yearly Income From Your Skills in 2024", fontsize=14
    # )
    ax.set_ylabel("Cumulative Expected Yearly Income", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a legend
    ax.legend(title="Skills", loc="upper left", frameon=True)

    # Save the plot to a file
    fig.savefig(
        save_path, bbox_inches="tight"
    )  # Save with tight layout to avoid clipping
    print(f"Plot saved to {save_path}")


# END OF CODE
