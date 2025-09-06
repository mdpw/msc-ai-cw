import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import os
import joblib
import json

# -----------------------------
# Paths
# -----------------------------
TRAINING_DIR = os.path.dirname(__file__)
MAIN_APP_DIR = os.path.abspath(os.path.join(TRAINING_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_APP_DIR, '..'))

DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'cinnamon_quality_dataset.csv')
MODELS_DIR = os.path.join(MAIN_APP_DIR, "backend", "model")
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)
df = df.drop(columns=['Sample_ID'])

# -----------------------------
# Handle null values
# -----------------------------
# Fill numerical NaNs with median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Drop rows with missing target
df = df.dropna(subset=['Quality_Label'])

# -----------------------------
# Remove noise/outliers using Z-score
# -----------------------------
z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
df = df[(z_scores < 3).all(axis=1)]

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop(columns=['Quality_Label'])
y = df['Quality_Label']
FEATURE_NAMES = X.columns.tolist()

# Save feature names
feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")
with open(feature_names_path, "w") as f:
    json.dump(FEATURE_NAMES, f)

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

# -----------------------------
# Label encoding
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Save label encoder
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

# -----------------------------
# Save training dataset for SHAP
# -----------------------------
train_orig_df = X_train.copy()
train_orig_df['Quality_Label'] = y_train
train_orig_path = os.path.join(MODELS_DIR, "train_for_shap.csv")
train_orig_df.to_csv(train_orig_path, index=False)

# -----------------------------
# Convert to PyTorch tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

# -----------------------------
# Handle class imbalance with WeightedRandomSampler
# -----------------------------
classes = np.unique(y_train_enc)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_enc)
sample_weights = np.array([class_weights[label] for label in y_train_enc])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# -----------------------------
# DataLoaders
# -----------------------------
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# -----------------------------
# Helper function for API inference
# -----------------------------
def preprocess_input(features):
    """
    Preprocess a single API request.
    features: list of numerical features
    """
    features_scaled = scaler.transform([features])
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    return input_tensor

# -----------------------------
# Exports
# -----------------------------
__all__ = [
    'X_train_tensor', 'y_train_tensor',
    'X_val_tensor', 'y_val_tensor',
    'X_test_tensor', 'y_test_tensor',
    'train_loader', 'val_loader', 'test_loader',
    'le', 'y_test_enc', 'X_test_scaled', 'scaler',
    'preprocess_input'
]
