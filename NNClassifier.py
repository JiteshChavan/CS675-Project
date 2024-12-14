import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the dataset
data_file = "ev_charging_patterns.csv"
df = pd.read_csv(data_file)

# Assign numerical values to User Type
user_type_mapping = {
    "Commuter": 0,
    "Casual Driver": 1,
    "Long-Distance Traveler": 2
}
df["User Type"] = df["User Type"].map(user_type_mapping)

# Check label tags
for i in range(df["User Type"].shape[0]):
    assert (df["User Type"][i] == 0 or df["User Type"][i] == 1 or df["User Type"][i] == 2), f"failed processing label tags"
print("> Processing of label tags successful.")

# Separate features and target
target = "User Type"
Y = df[target]

# Irrelevant Features to be dropped for modeling the classification
irrelevant_features = ["User ID", "Charging Station ID", "Charging Start Time", "Charging End Time", "User Type", "Vehicle Age (years)", "Temperature (°C)", "Charging Station Location", "Day of Week", "Time of Day"]
X = df.drop(columns=irrelevant_features)
print("Irrelevant features dropped successfully")

# Identify feature types
numeric_features = [
    "Battery Capacity (kWh)", "Energy Consumed (kWh)", 
    "Distance Driven (since last charge) (km)", "Charging Rate (kW)", "Charging Cost (USD)", "State of Charge (Start %)",
    "State of Charge (End %)", "Charging Duration (hours)"
]
categorical_features = ["Vehicle Model",  "Charger Type"]

# Label Encoding for categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Preprocessing: Handling missing values followed by standard scaling
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
    ("scaler", StandardScaler())  # Standard scaling
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing to the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, Y, test_size=0.2, random_state=42)

# Convert data to tensors for PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define the Neural Network Classifier
class NNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 64  # You can adjust this based on experimentation
output_dim = 3  # User Type has 3 classes

model = NNClassifier(input_dim, hidden_dim, output_dim)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training the model
num_epochs = 40  # You can adjust based on the data
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)
    print("Neural Network Classification Report:")
    print(classification_report(y_test_tensor, predicted))



