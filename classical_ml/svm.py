import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes up from /training
MODELS_DIR = os.path.join(BASE_DIR, "backend", "model")

# -----------------------
# Dataset path (root/dataset/processed)
# -----------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_path = os.path.join(project_root, 'dataset', 'processed', 'cinnamon_quality_dataset.csv')

# -----------------------------
# 1. Load the dataset
# -----------------------------
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=["Sample_ID", "Quality_Label"])
y = data["Quality_Label"]

# -----------------------------
# 2. Encode Target Labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# 4. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Hyperparameter Tuning with GridSearchCV
# -----------------------------
param_grid = {
    'C': [1, 10],
    'gamma': ['scale', 0.1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

print("🔍 Performing Grid Search for best parameters...")
grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# -----------------------------
# 6. Train Final SVM Model
# -----------------------------
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_scaled, y_train)

# -----------------------------
# 7. Evaluation
# -----------------------------
y_pred = best_svm.predict(X_test_scaled)

print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
