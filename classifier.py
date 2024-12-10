import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

for i in range (df["User Type"].shape[0]):
    assert (df["User Type"][i] == 0 or df["User Type"][i] == 1 or df["User Type"][i] == 2), f"failed processing label tags"

print ("> Procession of label tags successfull.")

# Separate features and target
target = "User Type"
Y = df[target]

# Irrelevant Features to be dropped for modeling the classfication (design choice, semantics, heuristics)
irrelevant_features = ["User ID", "Charging Station ID", "Charging Start Time", "Charging End Time", "User Type"]
X = df.drop(columns=irrelevant_features)
assert (X.shape[-1] == 15), f"failed dropping irrelevant features"
print ("Irrelevant features dropped succesfully")


# Identify feature types
numeric_features = [
    "Battery Capacity (kWh)", "Energy Consumed (kWh)", "Charging Duration (hours)",
    "Charging Rate (kW)", "Charging Cost (USD)", "State of Charge (Start %)",
    "State of Charge (End %)", "Distance Driven (since last charge) (km)",
    "Temperature (°C)", "Vehicle Age (years)"
]
categorical_features = ["Vehicle Model", "Charging Station Location", "Time of Day", "Day of Week", "Charger Type"]




print (X.shape, Y.shape)