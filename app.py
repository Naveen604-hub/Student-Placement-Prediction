import streamlit as st
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "placement_model.pkl"

st.title("Welcome to Student Placement Prediction Model")

st.write("Enter the student's details:")

cgpa = st.number_input("UG's CGPA", min_value=0.0, max_value=10.0, value=8.0)
tenth = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=88.0)
twelfth = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=86.0)
backlogs = st.number_input("No of Backlogs", min_value=0, value=0)
internships = st.number_input("Internships Completed", min_value=0, value=1)
projects = st.number_input("No of Projects Done", min_value=0, value=3)
skills = st.number_input("Skills Score", min_value=0, value=8)
aptitude = st.number_input("Aptitude Score", min_value=0, value=8)

if st.button("Predict Placement Status"):
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run 'python train_model.py' once in the project folder to generate placement_model.pkl.")
        st.stop()

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    new_student = [[
        cgpa,
        tenth,
        twelfth,
        backlogs,
        internships,
        projects,
        skills,
        aptitude
    ]]

    prediction = model.predict(new_student)

    if prediction[0] == 1:
        st.success("🎉 Prediction Results: Student is Placed!")
    else:
        st.error("Prediction Results: Student is not Placed Please Improve Skills and Prepare.")