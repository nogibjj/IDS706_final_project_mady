"""
This module implements the core functionality for the [app name] application.
It includes data processing, API routes, and interaction with external services.
"""

from flask import Flask, render_template, request, redirect, url_for
import os
from mylib.resume_summary import get_summary, get_top_skills
import pandas as pd
from gensim.models import Word2Vec
from mylib.figures import (
    plot_skill_demand_over_time,
    plot_skill_income_waterfall,
    plot_state_demand,
)

# Added because matplotlib messes up with debugging sometimes
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend


app = Flask(__name__)

DEFAULT_RESUME = "test_resume/oilworkerresume.pdf"  # Default test resume
UPLOAD_FOLDER = "uploads"  # Folder to save uploaded resumes
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PLOT_FOLDER = "static/fig"  # Folder to store generated plots
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOT_FOLDER"] = PLOT_FOLDER


@app.route("/", methods=["GET", "POST"])
def input_page():
    if request.method == "POST":
        # Handle file upload
        file = request.files["file"]
        if file and file.filename != "":
            # Save the uploaded file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
        else:
            # No file uploaded; use default resume
            file_path = DEFAULT_RESUME

        # Generate analysis using resume_summary functions
        summary = get_summary(file_path)
        top_skills = get_top_skills(summary)

        # Generate updated plots **before** redirecting
        jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")
        model = Word2Vec.load("data/skills_word2vec.model")

        # Generate the plots dynamically with unique names to avoid caching
        skill_demand_path = os.path.join(PLOT_FOLDER, "skill_demand_plot.png")
        state_demand_path = os.path.join(PLOT_FOLDER, "state_demand_plot.png")
        plot_skill_demand_over_time(
            jobpostDF=jobpostDF,
            skills_of_interest=top_skills,
            model=model,
            save_path=skill_demand_path,
        )
        plot_state_demand(
            skills_of_interest=top_skills,
            model=model,
            jobpostDF=jobpostDF,
            shapefile_path="data/tl_2024_us_state.shp",
            save_path=state_demand_path,
        )

        # Redirect to the analysis dashboard
        return redirect(
            url_for(
                "analysis_dashboard",
                file_path=file_path,
                skill_demand_path=url_for(
                    "static", filename=f"fig/skill_demand_plot.png"
                ),
                state_demand_path=url_for(
                    "static", filename=f"fig/state_demand_plot.png"
                ),
            )
        )

    return render_template("Resume_Input.html")


@app.route("/analysis", methods=["GET"])
def analysis_dashboard():
    # Get query parameters
    file_path = request.args.get("file_path", DEFAULT_RESUME)
    skill_demand_path = request.args.get(
        "skill_demand_path", url_for("static", filename="fig/skill_demand_plot.png")
    )
    state_demand_path = request.args.get(
        "state_demand_path", url_for("static", filename="fig/state_demand_plot.png")
    )

    # Perform analysis using resume_summary.py functions
    summary = get_summary(file_path)
    top_skills = get_top_skills(summary)

    return render_template(
        "Analysis_Dashboard.html",
        summary=summary,
        top_skills=top_skills,
        skill_demand_plot=skill_demand_path,
        state_demand_plot=state_demand_path,
    )


@app.route("/improvement", methods=["GET"])
def improvement_dashboard():
    file_path = request.args.get("file_path", DEFAULT_RESUME)
    return render_template("Improvement_Dashboard.html", file_path=file_path)


if __name__ == "__main__":
    app.run(debug=True)
