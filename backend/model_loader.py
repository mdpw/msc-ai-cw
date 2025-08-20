import torch
from model import ANN
from training.preprocess import X_train_tensor, le

# Infer input/output size from training data
input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = len(le.classes_)

def load_model(model_path="backend/model/best_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device
