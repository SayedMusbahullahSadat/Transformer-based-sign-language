import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and Preprocess the Dataset
# Load the dataset
data = pd.read_csv(r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\cleaned_landmarks_dataset.csv")

# Separate features and labels
X = data.drop(columns=["label"])  # Features
y = data["label"]  # Labels

# Normalize the feature values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)



# Step 2: Define the Neural Network Model
class SignLanguageNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignLanguageNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 64)         # Hidden layer to hidden layer
        self.fc3 = nn.Linear(64, num_classes)  # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Step 3: Initialize Model, Loss Function, and Optimizer
# Define model parameters
input_size = X_train_tensor.shape[1]  # Number of features
num_classes = len(label_encoder.classes_)  # Number of unique labels

# Initialize the model, loss function, and optimizer
model = SignLanguageNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # Optimizer

# Step 4: Train the Model
# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 5: Evaluate the Model
# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)

# Calculate accuracy
accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 6: Save and Load the Model
# Save the model
torch.save(model.state_dict(), "sign_language_nn.pth")

# Load the model
loaded_model = SignLanguageNN(input_size, num_classes)
loaded_model.load_state_dict(torch.load("sign_language_nn.pth"))
loaded_model.eval()

# Step 7: Predict on New Data
# Example: Predict on a single test sample
new_sample = X_test_tensor[0].unsqueeze(0)  # Add batch dimension
with torch.no_grad():
    prediction = loaded_model(new_sample)
    _, predicted_label = torch.max(prediction, 1)

# Decode the predicted label
print("Predicted Sign:", label_encoder.inverse_transform(predicted_label.numpy()))
