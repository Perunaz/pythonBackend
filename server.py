import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from model import Model  # Import your model definition
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

model_instance = Model()

# Create LabelEncoders for categorical columns
label_encoders = {}
categorical_columns = ['gender', 'smoking_history']

for col in categorical_columns:
    le = LabelEncoder()
    label_encoders[col] = le

@app.route("/image", methods=["GET"])
def handle_image_request():
    return send_file("./temp.png", mimetype="image/png")

# Endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def prediction():
    # Retrieve the input data from the POST request
    data = request.get_json()

    # Convert the input data to a dictionary
    input_data = {}
    for key, value in data.items():
        input_data[key] = [value]

    # Assuming input_data is in the same order as your original dataset
    input_data = pd.DataFrame(input_data)

    # Encode categorical features
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        input_data[col] = le.fit_transform(input_data[col])
        label_encoders[col] = le

    # Convert the input data into a NumPy array
    data_array = np.array(input_data)

    # Make the prediction
    prediction, certainty = model_instance.predict(data_array)

    certainty_value = certainty[0]
    prediction_value = float(prediction[0])

    response = {
        "predicted_class": prediction_value,
        "certainty": certainty_value
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)