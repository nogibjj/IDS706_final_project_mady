from flask import Flask, render_template, request, redirect, url_for
import os
from mylib.resume_summary import get_summary, get_top_skills

app = Flask(__name__)

DEFAULT_RESUME = "test_resume/oilworkerresume.pdf"  # Default test resume
UPLOAD_FOLDER = "uploads"  # Folder to save uploaded resume
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PLOT_FOLDER = "fig"  # Folder to find plots generated
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOT_FOLDER"] = PLOT_FOLDER

@app.route("/", methods=["GET", "POST"])
def input_page():
    if request.method == "POST":
        # Handle file uploaded
        file = request.files["file"]

        if file and file.filename != "":
            # Save the uploaded file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
        else:
            # No file uploaded, use default resume
            file_path = DEFAULT_RESUME

        # Generate analysis using resume_summary functions
        summary = get_summary(file_path)
        top_skills = get_top_skills(summary)

        # Redirect to analysis dashboard and pass data
        return render_template(
            "Analysis_Dashboard.html", 
            summary=summary, 
            top_skills=top_skills, 
            plot_path=url_for('static', filename='fig/skill_demand_plot.png'),
            file_path=file_path  # Pass the file path to the analysis page
        )

    return render_template("Resume_Input.html")


@app.route("/analysis", methods=["GET"])
def analysis_dashboard():
    # Get file_path from query parameters
    file_path = request.args.get('file_path', DEFAULT_RESUME)  # Use default if not provided

    # Perform analysis using resume_summary.py functions
    summary = get_summary(file_path)
    top_skills = get_top_skills(file_path)

    return render_template(
        "Analysis_Dashboard.html", summary=summary, top_skills=top_skills, file_path=file_path
    )

@app.route("/improvement", methods=["GET"])
def improvement_dashboard():
    # Get file_path from query parameters
    file_path = request.args.get('file_path', DEFAULT_RESUME)  # Get the file_path from query params

    # You can add additional analysis here if needed
    return render_template("Improvement_Dashboard.html", file_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
