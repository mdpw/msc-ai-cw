# baseline.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# ---------------------------
# Import preprocessed data
# ---------------------------
from main_app.backend.base_model import BaseANN
from main_app.training.preprocess import (
    X_train_tensor, y_train_tensor,
    X_val_tensor, y_val_tensor,
    X_test_tensor, y_test_tensor,
    train_loader, val_loader, test_loader,
    le, X_test_scaled
)

# ---------------------------
# Hyperparameters (Baseline)
# ---------------------------
input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = len(le.classes_)
learning_rate = 0.01
num_epochs = 50

# ---------------------------
# Model, loss, optimizer
# ---------------------------
model = BaseANN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Directory setup for saving best model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/model'))
#model_dir = os.path.join(os.path.dirname(__file__), "backend", "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "base_model.pth")

# ---------------------------
# Figure Saving Directory
# ---------------------------
fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "base_model_figs"))
os.makedirs(fig_dir, exist_ok=True)

def save_fig(fig, name):
    fig_path = os.path.join(fig_dir, name)
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Saved: {fig_path}")
# ---------------------------
# Training Loop
# ---------------------------
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("Starting baseline training...\n")
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

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save the best model based on validation loss
torch.save(model.state_dict(), model_path)

print(f"\nTraining complete. Base model saved to: {model_path}")
print("\nBaseline training complete.")

# ---------------------------
# Plot Training History
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve - Base Model')
axes[0].legend()

axes[1].plot(train_accuracies, label='Train Accuracy')
axes[1].plot(val_accuracies, label='Val Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curve - Base Model')
axes[1].legend()

plt.tight_layout()
save_fig(fig, "training_history_base.png")
plt.close(fig)


# ---------------------------
# Final Evaluation
# ---------------------------
print("\nEvaluating model on test set...")

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
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix - Base Model")
save_fig(fig, "confusion_matrix_base.png")
plt.close(fig)


# ---------------------------
# Classification Report Heatmap
# ---------------------------
report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="Greens", fmt=".2f", ax=ax)
ax.set_title("Classification Report Heatmap - Base Model")
save_fig(fig, "classification_report_base.png")
plt.close(fig)

# ---------------------------
# ROC & Precision-Recall Curves
# ---------------------------
y_true_bin = label_binarize(y_true, classes=range(output_size))
y_score = model(torch.tensor(X_test_scaled, dtype=torch.float)).detach().numpy()

# ROC curves
fig, ax = plt.subplots(figsize=(10, 7))
for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Base Model")
ax.legend()
save_fig(fig, "roc_curves_base.png")
plt.close(fig)


# Precision-Recall curves
fig, ax = plt.subplots(figsize=(10, 7))
for i, class_name in enumerate(le.classes_):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
    ax.plot(recall, precision, label=f"{class_name} (AP={ap:.2f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves - Base Model")
ax.legend()
save_fig(fig, "precision_recall_curves_base.png")
plt.close(fig)

