import os
import torch
import pandas as pd
from backend.model import ANN
from training.preprocess import scaler, le, X_train_tensor

# Prediction function
def predict_quality(
    model_class,            # Your ANN class
    model_path,             # Path to saved model weights, e.g., "backend/model/best_model.pth"
    scaler,                 # Fitted StandardScaler used during training
    label_encoder,          # Fitted LabelEncoder used during training
    new_data: pd.DataFrame  # New samples with same features as training data
):
    # 1. Preprocess new data
    new_data_scaled = scaler.transform(new_data)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # 2. Load model
    input_size = X_train_tensor.shape[1]
    hidden_size = 64
    output_size = len(le.classes_)
    dropout_rate = 0.2
    
    model = ANN(
    input_size=input_size,
    hidden_size=hidden_size,
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
    return predicted_labels, probabilities

# Example usage - This should return Low
new_samples = pd.DataFrame([
    {
        "Moisture": 13.66064476,
        "Ash": 7.418681347266111,
        "Volatile_Oil": 0.8231678150936274,
        "Acid_Insoluble_Ash": 0.5084406430686356,
        "Chromium": -0.092484138,
        "Coumarin": -0.117050129
    }
])

							
			


# Model path
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/model/best_model.pth'))
os.makedirs(os.path.dirname(model_path), exist_ok=True)

predicted_labels, probabilities = predict_quality(
    model_class=ANN,
    model_path=model_path,
    scaler=scaler,
    label_encoder=le,
    new_data=new_samples
)

print("Predicted labels:", predicted_labels)
print("Predicted probabilities:\n", probabilities)
