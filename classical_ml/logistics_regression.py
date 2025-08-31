

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# Drop Sample_ID since it's not a feature
X = data.drop(columns=["Sample_ID", "Quality_Label"])
y = data["Quality_Label"]

# -----------------------------
# 2. Encode the target labels
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
# 5. Logistic Regression Model
# -----------------------------
log_reg = LogisticRegression(
    multi_class='multinomial',  # for multi-class classification
    solver='lbfgs',
    max_iter=10000,
    random_state=42
)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = log_reg.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
