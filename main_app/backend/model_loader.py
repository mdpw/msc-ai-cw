import torch
from .best_model import ANN
from training.preprocess import X_train_tensor, le
from config import CONFIG

# Infer input/output size from training data
input_size = X_train_tensor.shape[1]
output_size = len(le.classes_)

hidden1_size = CONFIG["best_model"]["model"]["hidden1_size"]
hidden2_size = CONFIG["best_model"]["model"]["hidden2_size"]
dropout_rate = CONFIG["best_model"]["model"]["dropout_rate"]

def load_model(model_path="backend/model/best_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ANN(
        input_size=input_size,
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device