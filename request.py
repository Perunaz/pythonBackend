import requests
import json

# Sample input data
input_data = {
    'gender': 'Male',
    'age': 50.0,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'former',
    'bmi': 37.16,
    'HbA1c_level': 9.0,
    'blood_glucose_level': 159
}

# Make a POST request to the Flask app
url = "http://localhost:5000/predict"
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=input_data)

# Print the response
print(response.json())