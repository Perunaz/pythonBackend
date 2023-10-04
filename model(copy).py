import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# Step 1: Load and preprocess the dataset
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical features
label_encoders = {}
categorical_columns = ['gender', 'smoking_history']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(['diabetes'], axis=1)  # Features
y = data['diabetes']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32).to(device)
        self.y = torch.tensor(y.values, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 3: Define the model
class ClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(1024, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.softmax(x)
        return x

input_dim = X_train.shape[1]
output_dim = len(data['diabetes'].unique())
model = ClassificationModel(input_dim, output_dim)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Implement learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 10

# Training loop
# Training loop
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    test_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Update learning rate
    scheduler.step()

# Plotting the training loss and test accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate the model
model.eval().to(device)
with torch.no_grad():
    # Evaluate the test dataset
    correct = 0
    total = 0

    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

pickle.dump(model, open('model.pkl', 'wb'))

# Loading model to compare the results
model1 = pickle.load(open('model.pkl', 'rb'))

# Create an input tensor for prediction
input_data = pd.DataFrame({
    'gender': ['Male'],
    'age': [50.0],
    'hypertension': [0],
    'heart_disease': [0],
    'smoking_history': ['former'],
    'bmi': [37.16],
    'HbA1c_level': [9.0],
    'blood_glucose_level': [159]
})

# Encode categorical features
for col in categorical_columns:
    le = label_encoders[col]
    input_data[col] = le.transform(input_data[col])

# Scale numerical features (if you used any scaling during training)
# Example assuming Min-Max scaling:
# input_data[numerical_columns] = min_max_scaler.transform(input_data[numerical_columns])

# Convert the input data to a tensor and move it to the appropriate device
input_tensor = torch.tensor(input_data.values, dtype=torch.float32).to(device)

# Ensure the model is in evaluation mode
model1.eval()

# Forward pass (prediction)
with torch.no_grad():
    predictions = model1(input_tensor)

# Convert the predictions to probabilities and select the class with the maximum probability
probabilities = torch.softmax(predictions, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

# Get the probability associated with the predicted class
predicted_class_probability = probabilities[0][predicted_class].item()

# Print the predicted class
print("Predicted class:", predicted_class, "Probability:", predicted_class_probability)
