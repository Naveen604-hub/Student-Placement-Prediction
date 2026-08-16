import pickle

# Load the trained model
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)

# Student details
cgpa = float(input("Enter CGPA: "))
tenth = float(input("Enter 10th percentage: "))
twelfth = float(input("Enter 12th percentage: "))
internship = int(input("Has internship? (1 = Yes, 0 = No): "))
projects = int(input("Enter number of projects: "))
technical = float(input("Enter technical skills rating (1-10): "))
communication = float(input("Enter communication skills rating (1-10): "))
backlogs = int(input("Enter number of backlogs: "))

# Prepare the input
student = [[
    cgpa,
    tenth,
    twelfth,
    internship,
    projects,
    technical,
    communication,
    backlogs
]]

# Make prediction
prediction = model.predict(student)

# Display result
if prediction[0] == 1:
    print("\n🎉 Prediction Results: Student is Placed!")
else:
    print("\nPrediction Results: Student is not Placed Please Improve Skills and Prepare.")