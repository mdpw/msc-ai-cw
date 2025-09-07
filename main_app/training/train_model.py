import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Import preprocessed data and ANN
# ---------------------------
from main_app.backend.model import ANN
from main_app.training.preprocess import (
    X_train_tensor, y_train_tensor,
    X_val_tensor, y_val_tensor,
    X_test_tensor, y_test_tensor,
    train_loader, val_loader, test_loader,
    le, y_test_enc, X_test_scaled, scaler
)

# ---------------------------
# Hyperparameters & Setup
# ---------------------------
input_size = X_train_tensor.shape[1]
hidden1_size = 128
hidden2_size = 64
output_size = len(le.classes_)
dropout_rate = 0.2
learning_rate = 0.001
num_epochs = 150
patience = 10
weight_decay = 1e-4
factor = 0.5

# Model, loss, optimizer, and scheduler
model = ANN(
    input_size=input_size,
    hidden1_size=hidden1_size,
    hidden2_size=hidden2_size,
    output_size=output_size,
    dropout_rate=dropout_rate
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

# Directory setup for saving best model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/model'))
#model_dir = os.path.join(os.path.dirname(__file__), "backend", "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_model.pth")

# ---------------------------
# Training Loop
# ---------------------------
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
early_stop_counter = 0

print("Starting training...\n")
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nTraining complete. Best model saved to: {model_path}")

# ---------------------------
# Plot Training History
# ---------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()


# ---------------------------
# Final Evaluation & Visualization
# ---------------------------
print("\nEvaluating best model on test set...")

# Load the best model
model.load_state_dict(torch.load(model_path))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# Classification Report Heatmap
# ---------------------------
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="Greens", fmt=".2f")
plt.title("Classification Report Heatmap")
plt.show()

# ---------------------------
# ROC & Precision-Recall Curves
# ---------------------------
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Binarize labels for multi-class curves
y_true_bin = label_binarize(y_true, classes=range(output_size))
y_score = model(torch.tensor(X_test_scaled, dtype=torch.float)).detach().numpy()

# ROC curves
plt.figure(figsize=(10, 7))
for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Precision-Recall curves
plt.figure(figsize=(10, 7))
for i, class_name in enumerate(le.classes_):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=f"{class_name} (AP={ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.show()

# ---------------------------
# Learning Rate Schedule (Optional)
# ---------------------------
# Note: AdamW keeps lr in optimizer param_groups
lrs = [group['lr'] for group in optimizer.param_groups]
plt.plot(lrs, marker='o')
plt.title("Learning Rate Schedule")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.show()
