# START OF CODE

from flask import Flask, render_template, request, redirect, url_for
import os
from mylib.resume_summary import get_summary, get_top_skills
import pandas as pd
from gensim.models import Word2Vec
from mylib.figures import (
    plot_skill_demand_over_time,
    plot_state_demand,
    plot_skill_income_waterfall,
)
import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)

DEFAULT_RESUME = "test_resume/oilworkerresume.pdf"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PLOT_FOLDER = "static/fig"
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOT_FOLDER"] = PLOT_FOLDER


@app.route("/", methods=["GET", "POST"])
def input_page():
    if request.method == "POST":
        file = request.files["file"]
        file_path = (
            os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            if file
            else DEFAULT_RESUME
        )
        if file and file.filename:
            file.save(file_path)

        summary = get_summary(file_path)
        top_skills = get_top_skills(summary)

        jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")
        model = Word2Vec.load("data/skills_word2vec.model")

        plot_skill_demand_over_time(
            jobpostDF,
            top_skills,
            model,
            os.path.join(PLOT_FOLDER, "skill_demand_plot.png"),
        )
        plot_state_demand(
            top_skills,
            model,
            jobpostDF,
            save_path=os.path.join(PLOT_FOLDER, "state_demand_plot.png"),
        )
        plot_skill_income_waterfall(
            top_skills,
            jobpostDF,
            save_path=os.path.join(PLOT_FOLDER, "skill_income_waterfall.png"),
        )

        return redirect(url_for("analysis_dashboard", file_path=file_path))

    return render_template("Resume_Input.html")


@app.route("/analysis", methods=["GET"])
def analysis_dashboard():
    file_path = request.args.get("file_path", DEFAULT_RESUME)
    summary = get_summary(file_path)
    top_skills = get_top_skills(summary)

    return render_template(
        "Analysis_Dashboard.html",
        summary=summary,
        top_skills=top_skills,
        skill_demand_plot=url_for("static", filename="fig/skill_demand_plot.png"),
        state_demand_plot=url_for("static", filename="fig/state_demand_plot.png"),
        skill_income_waterfall=url_for(
            "static", filename="fig/skill_income_waterfall.png"
        ),
    )


@app.route("/improvement", methods=["GET"])
def improvement_dashboard():
    file_path = request.args.get("file_path", DEFAULT_RESUME)
    return render_template("Improvement_Dashboard.html", file_path=file_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

# END OF CODE
