import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import math


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
irrelevant_features = ["User ID", "Charging Station ID", "Charging Start Time", "Charging End Time", "User Type", "Vehicle Age (years)", "Temperature (°C)", "Charging Station Location", "Day of Week", "Time of Day"]
i_f2 = []
irrelevant_features+=i_f2
X = df.drop(columns=irrelevant_features)
print ("Irrelevant features dropped succesfully")


# Identify feature types
numeric_features = [
    "Battery Capacity (kWh)", "Energy Consumed (kWh)", 
    "Distance Driven (since last charge) (km)", "Charging Rate (kW)", "Charging Cost (USD)", "State of Charge (Start %)",
    "State of Charge (End %)", "Charging Duration (hours)"
]
categorical_features = ["Vehicle Model",  "Charger Type"]

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Pre processing : Handling missing values followed by standard scaling
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
    ("scaler", StandardScaler())  # Standard scaling
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values
])

# Combine preprocessors in a column transformer
# not that transformer.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing to the data
X_preprocessed = preprocessor.fit_transform(X)

for feature in categorical_features:
    print(f"{feature}: {df[feature].nunique()} unique values")

# 19 additional features created to one hot represent categorical features
# total features 34
print(f"Preprocessed feature shape: {X_preprocessed.shape}")
print(f"Target shape: {Y.shape}")

print (X.shape, Y.shape)


# List of numeric features
numeric_features = df.select_dtypes(include=['number']).columns

# Set number of plots per figure
plots_per_figure = 4
num_features = len(numeric_features)
num_figures = math.ceil(num_features / plots_per_figure)
for i in range(num_figures):
    # Create a new figure for each set of 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    
    # Iterate through the current set of features
    for j in range(plots_per_figure):
        feature_idx = i * plots_per_figure + j
        if feature_idx >= num_features:  # Break if we've plotted all features
            break
        feature = numeric_features[feature_idx]
        
        # Plot on the current axis
        sns.histplot(df[feature], kde=True, bins=30, color='blue', ax=axes[j])
        axes[j].set_title(f"Distribution of {feature}")
        axes[j].set_xlabel(feature)
        axes[j].set_ylabel("Frequency")
    
    # Turn off unused axes
    for j in range(plots_per_figure):
        if i * plots_per_figure + j >= num_features:
            axes[j].axis('off')
    
    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

# Target variable distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=Y, palette="viridis")
plt.title("Target Variable Distribution (User Type)")
plt.xlabel("User Type")
plt.ylabel("Count")
plt.show()

# Correlation matrix of numeric features
plt.figure(figsize=(12, 8))
corr_matrix = df[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# Correlation of features with the target variable
#correlations_with_target = df.corr()[target].sort_values(ascending=False)


#### NO IMBLANCE IN TARGET VARIBLE!! as seen from the graph

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y, test_size=0.2, random_state=42)
print(type(X_train))
print(type(X_train))

print (X_test.shape, X_train.shape)
#import sys; sys.exit(0)

# Define models with probability estimation enabled for SVM
log_reg = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(max_depth=20, min_samples_split=10, min_samples_leaf=4, random_state=42, class_weight="balanced")
svm = SVC(random_state=42, probability=True)  # Enable probability estimates

# Now you can fit the models as before


# hyperparameter grids
log_reg_param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2']  # Type of regularization
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Max depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required at a leaf node
}

svm_param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient for RBF
    'kernel': ['rbf']  # Using RBF kernel
}

# Perform GridSearchCV for Logistic Regression
log_reg_grid_search = GridSearchCV(estimator=log_reg, param_grid=log_reg_param_grid, 
                                   cv=5, n_jobs=-1, scoring='accuracy')
log_reg_grid_search.fit(X_train, Y_train)

# Perform GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, 
                              cv=5, n_jobs=-1, scoring='accuracy')
rf_grid_search.fit(X_train, Y_train)

svm_grid_search = GridSearchCV(estimator=svm, param_grid=svm_param_grid, 
                               cv=5, n_jobs=-1, scoring='accuracy')
svm_grid_search.fit(X_train, Y_train)


# Best models and their hyperparameters
print("Best Logistic Regression Hyperparameters:", log_reg_grid_search.best_params_)
print("Best Random Forest Hyperparameters:", rf_grid_search.best_params_)

# Evaluate the models on the test set
log_reg_best = log_reg_grid_search.best_estimator_
rf_best = rf_grid_search.best_estimator_
svm_best = svm_grid_search.best_estimator_

log_reg_predictions = log_reg_best.predict(X_test)
rf_predictions = rf_best.predict(X_test)
svm_predictions = svm_best.predict(X_test)

print("Best Logistic Regression Hyperparameters:", log_reg_grid_search.best_params_)
print("Best Random Forest Hyperparameters:", rf_grid_search.best_params_)
print("Best SVM Hyperparameters:", svm_grid_search.best_params_)

# Print classification reports
print("\nLogistic Regression Classification Report:\n", classification_report(Y_test, log_reg_predictions))
print("Random Forest Classification Report:\n", classification_report(Y_test, rf_predictions))
print("SVM Classification Report:\n", classification_report(Y_test, svm_predictions))

# Evaluate models using cross-validation
log_reg_cv_score = cross_val_score(log_reg_best, X_train, Y_train, cv=5, scoring='accuracy').mean()
rf_cv_score = cross_val_score(rf_best, X_train, Y_train, cv=5, scoring='accuracy').mean()
svm_cv_score = cross_val_score(svm_best, X_train, Y_train, cv=5, scoring='accuracy').mean()

print("\nLogistic Regression Cross-Validation Score: ", log_reg_cv_score)
print("Random Forest Cross-Validation Score: ", rf_cv_score)
print("SVM Cross-Validation Score: ", svm_cv_score)


import shap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have the predictions from the best models:
log_reg_predictions = log_reg_best.predict(X_test)
rf_predictions = rf_best.predict(X_test)
svm_predictions = svm_best.predict(X_test)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrix for each model
plot_confusion_matrix(Y_test, log_reg_predictions, "Logistic Regression Confusion Matrix")
plot_confusion_matrix(Y_test, rf_predictions, "Random Forest Confusion Matrix")
plot_confusion_matrix(Y_test, svm_predictions, "SVM Confusion Matrix")


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

def plot_auc_roc(y_true, y_pred_prob, model_name):
    # Use LabelBinarizer to binarize the labels (important for multi-class AUC)
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    
    # For multi-class, roc_auc_score expects y_pred_prob to have probabilities for all classes
    auc = roc_auc_score(y_true_bin, y_pred_prob, multi_class="ovr")  # One-vs-Rest AUC for multi-class
    
    # Plot ROC curve for each class
    n_classes = y_true_bin.shape[1]
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.2f})")
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"AUC-ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

# Now you can pass the model probabilities to plot AUC-ROC curves
log_reg_probs = log_reg_best.predict_proba(X_test)
rf_probs = rf_best.predict_proba(X_test)
svm_probs = svm_best.predict_proba(X_test)  # Ensure you're using predict_proba()

# Plot AUC-ROC for each model
plot_auc_roc(Y_test, log_reg_probs, "Logistic Regression AUC-ROC Curve")
plot_auc_roc(Y_test, rf_probs, "Random Forest AUC-ROC Curve")
plot_auc_roc(Y_test, svm_probs, "SVM AUC-ROC Curve")


# SHAP values (using Random Forest for this example)
explainer = shap.TreeExplainer(rf_best)  # Use an appropriate model explainer
shap_values = explainer.shap_values(X_test)

explainer = shap.Explainer(rf_best)
shap_values = explainer.shap_values(X_test)

# Check the shape of shap_values to ensure valid indexing
num_samples, num_features = shap_values[0].shape  # For multi-class, shap_values will be a list, one for each class
print(f"SHAP values shape for class 0: {shap_values[0].shape}")  # Sample shape for class 0

# Let's iterate over samples and features
for index in range(num_features):
    for inds in range(num_samples):
        # Ensure indices are within valid range
        if inds < num_samples and index < num_features:
            shap_ref = shap_values[0][inds, index]  # Accessing SHAP value for class 0, sample `inds`, feature `index`
            print(f"SHAP value for sample {inds}, feature {index}: {shap_ref}")
        else:
            print(f"Invalid indices: sample {inds}, feature {index}")


# Plot SHAP summary plot for feature importance
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
print ('herererererer')

