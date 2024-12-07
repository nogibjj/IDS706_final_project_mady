import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.generativeai import GenerativeModel

pd.set_option("mode.copy_on_write", True)


load_dotenv()
# Directly set your API key
api_key = os.environ["API_KEY"]
# Replace with your actual API key

# Configure the API key
genai.configure(api_key=api_key)

gemini_client = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a helpful assistant and an expert at reviewing resumes with years of experience in giving constructive feedback on professional's resumes",
)


def get_unique_skills():
    jobpostDF = pd.read_parquet("data/jobpostDF_subset.parquet")
    unique_skills = {}
    # go through each row of the job posting to identify the unique skills for Gemini to consider down the line
    for row in jobpostDF["matched_skills"]:
        for skill in row:
            unique_skills[skill] = 1
    unique_skill_df = pd.DataFrame(unique_skills.keys())
    # write into a txt file, no need to run this function again
    unique_skill_df.to_csv("data/skill.txt", index=False, sep=" ")


def get_summary(pdf_path, model=genai.GenerativeModel("gemini-1.5-flash")):
    """input the file path for the pdf cv as argument and the function will
    give a summary of the resume and the top five related skills with the resume"""
    # upload the pdf (the cv) and the skill txt to be processed for Gemini
    sample_pdf = genai.upload_file(pdf_path)
    skill_list = genai.upload_file("data/skill.txt")
    # detailed prompt to send to gemini on what we want and how we want the response
    response = model.generate_content(
        [
            "Give me three concise sentence summary of this resume,"
            "one sentence of overall what the resume focuses on as well as what type of career this person looks like they are targeting, "
            "one sentence on what the resume does a good job on and one sentence on what could be improved on. "
            "Can you also give me five top related from this skill list attached. Do not give me skills that are not in the list attached, "
            "the skills you give have to be exactly the same as the ones in the list and the skills MUST be only one word! "
            "Title the skills list as 'Your resume highlights these following skills:'  "
            "Do not use a colon in your response otherthan when you list the skills.",
            sample_pdf,
            skill_list,
        ]
    )
    return response.text


def get_top_skills(summary_text):
    """Takes the resume summary from Gemini as an input and ouputs a list of skills to be send to the NLP model"""
    top_skills = summary_text.split(":")[1]
    top_skills = top_skills.splitlines()
    skill_list = []
    for skill in top_skills:
        skill = skill.lower()
        if skill.isalpha():
            skill_list.append(skill)
    return skill_list


# if __name__ == "__main__":
#     get_summary(
#         "test_resume/example_resume.pdf"
#     )  # need to update with a real pdf path for this to work, just have a test here to show that it works
#     resume_summary = get_summary("test_resume/example_resume.pdf")
#     get_top_skills(resume_summary)

if __name__ == "__main__":
    # Default to the test resume for standalone testing
    resume_path = "test_resume/example_resume.pdf"

    try:
        summary = get_summary(resume_path)
        print("Resume Summary:\n", summary)

        skills = get_top_skills(summary)
        print("Top Skills:\n", skills)
    except Exception as e:
        print(f"Error: {e}")
