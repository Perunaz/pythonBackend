from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
from model import ClassificationModel  # Import your model definition
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
input_dim = 8
output_dim = 2
categorical_columns = ['gender', 'smoking_history']

model = ClassificationModel(input_dim, output_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
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

    # Convert the input data to a tensor and move it to the appropriate device
    input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

    # Forward pass (prediction)
    with torch.no_grad():
        predictions = model(input_tensor)

    # Convert the predictions to probabilities using softmax
    probabilities = torch.softmax(predictions, dim=1)

    # Get the predicted class and its associated probability
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_class_probability = probabilities[0][predicted_class].item()

    # Prepare the response
    response = {
        "predicted_class": predicted_class,
        "certainty": predicted_class_probability
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, ssl_context=('/etc/ssl/certs/selfsigned.crt', '/etc/ssl/private/selfsigned.key'))
