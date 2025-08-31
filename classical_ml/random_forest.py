import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Features and target
X = data.drop(columns=["Sample_ID", "Quality_Label"])
y = data["Quality_Label"]

# Encode labels (Low, Medium, High → 0, 1, 2)
le = LabelEncoder()
y = le.fit_transform(y)

# ----------------------
# 2. Split Data
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 3. Scale Features
# ----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------
# 4. Train Random Forest
# ----------------------
rf = RandomForestClassifier(
    n_estimators=300,      # number of trees
    max_depth=15,         # control depth for better generalization
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ----------------------
# 5. Evaluate Model
# ----------------------
y_pred = rf.predict(X_test)

print("\n--- Random Forest Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
