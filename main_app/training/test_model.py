import os
import torch
import pandas as pd
from main_app.backend.best_model import ANN
from main_app.training.preprocess import scaler, le, X_train_tensor
from config import CONFIG
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Prediction function
# ---------------------------
def predict_quality(
    model_path,             # Path to saved model weights, e.g., "backend/model/best_model.pth"
    scaler,                 # Fitted StandardScaler used during training
    label_encoder,          # Fitted LabelEncoder used during training
    new_data: pd.DataFrame  # New samples with same features as training data
):
    # 1. Preprocess new data
    new_data_scaled = scaler.transform(new_data)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # 2. Load model with best hyperparameters
    input_size = X_train_tensor.shape[1]    
    output_size = len(le.classes_)

    hidden1_size = CONFIG["model"]["hidden1_size"]
    hidden2_size = CONFIG["model"]["hidden2_size"]
    dropout_rate = CONFIG["model"]["dropout_rate"]

    model = ANN(
        input_size=input_size,
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode

    # 3. Make predictions
    with torch.no_grad():
        outputs = model(new_data_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_indices = torch.max(outputs, 1)

    # 4. Convert indices to labels
    predicted_labels = label_encoder.inverse_transform(predicted_indices.numpy())
    return predicted_labels, probabilities.numpy()

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    new_samples = pd.DataFrame([
        {
            "Moisture": 7.5736,
            "Ash": 1.3275,
            "Volatile_Oil": 2.0913,
            "Acid_Insoluble_Ash": 0.1619,
            "Chromium": 0.4304,
            "Coumarin": 0.3629,
            "Fiber": 10.7925,
            "Density": 0.9129,
            "Oil_Content": 6.0226,
            "Resin": 2.3582,
            "Pesticide_Level": 0.0132,
            "PH_Value": 5.9011
        }
    ])

    # Model path
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/model/best_model.pth'))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    predicted_labels, probabilities = predict_quality(
        model_path=model_path,
        scaler=scaler,
        label_encoder=le,
        new_data=new_samples
    )

    print("Predicted labels:", predicted_labels)
    print("Predicted probabilities:\n", probabilities)
