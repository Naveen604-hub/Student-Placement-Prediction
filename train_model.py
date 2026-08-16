from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "placement_model.pkl"

# Load dataset
data = pd.read_csv(BASE_DIR / "dataset.csv")

# Features
X = data.drop("Placement", axis=1)

# Target
y = data["Placement"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the ML model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Student Placement Prediction Model")
print("-----------------------------------")
print("Model trained successfully!")
print("Accuracy:", accuracy)

# Save the trained model
with MODEL_PATH.open("wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

# Test with a new student
new_student = [[8.2, 88, 86, 1, 3, 8, 8, 0]]

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: Student is likely to be PLACED")
else:
    print("Prediction: Student is likely NOT to be PLACED")