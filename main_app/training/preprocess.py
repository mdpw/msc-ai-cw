import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import joblib
import os
import json


# main_app folder is current __file__ location
# Current folder is main_app/training
TRAINING_DIR = os.path.dirname(__file__)
# main_app folder
MAIN_APP_DIR = os.path.abspath(os.path.join(TRAINING_DIR, '..'))
# Project root (one level above main_app)
PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_APP_DIR, '..'))

# Correct dataset path
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'cinnamon_quality_dataset.csv')

# Backend model folder
MODELS_DIR = os.path.join(MAIN_APP_DIR, "backend", "model")
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# Load and preprocess dataset
# -----------------------
df = pd.read_csv(DATASET_PATH)
df = df.drop(columns=['Sample_ID'])
X = df.drop(columns=['Quality_Label'])
y = df['Quality_Label']

FEATURE_NAMES = X.columns.tolist()  # Save feature names

# Save feature names
feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")
with open(feature_names_path, "w") as f:
    json.dump(FEATURE_NAMES, f)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Save the label encoder
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# Helper function for API inference
def preprocess_input(features):
    """
    Preprocess a single API request.
    features: list of numerical features
    """
    features_scaled = scaler.transform([features])
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    return input_tensor

__all__ = [
    'X_train_tensor', 'y_train_tensor',
    'X_val_tensor', 'y_val_tensor',
    'X_test_tensor', 'y_test_tensor',
    'train_loader', 'val_loader', 'test_loader',
    'le', 'y_test_enc', 'X_test_scaled', 'scaler',
    'preprocess_input'
]
