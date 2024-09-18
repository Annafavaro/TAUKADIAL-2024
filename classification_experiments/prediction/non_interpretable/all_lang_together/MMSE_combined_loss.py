import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

class MMSEPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_bins):
        super(MMSEPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_mmse = nn.Linear(hidden_size, 1)  # Output layer for predicting MMSE score directly
        self.fc2_bins = nn.Linear(hidden_size, num_bins)  # Output layer for predicting binned MMSE interval
        # 10 = num of bins

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mmse_score = self.fc2_mmse(x)
        bins_logits = self.fc2_bins(x)
        bins_probabilities = F.softmax(bins_logits, dim=1)
        return mmse_score, bins_logits, bins_probabilities

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_true_mmse, y_true_bins, mmse_pred, bins_logits):
        # Calculate mean squared error for MMSE score prediction
        mse_loss = F.mse_loss(mmse_pred, y_true_mmse.float().unsqueeze(1))
        print(mse_loss)
        # Calculate cross-entropy loss for binned MMSE interval prediction
        ce_loss = F.cross_entropy(bins_logits, y_true_bins.long(), reduction='sum')
        print(ce_loss)
        # Combine both losses
        loss = (1 / torch.sum(y_true_bins)) * (mse_loss - ce_loss)

        return loss


# Define hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32
input_size = 10
hidden_size = 40
num_bins = 10
# Define your model
model = MMSEPredictionModel(input_size, hidden_size, num_bins)

# Define your loss function
loss_fn = CustomLoss()

# Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Assuming you have initialized your model, loss function, and data loaders

X_train = torch.randn(100, 10)  # Example input data with 100 samples and 10 features
y_true_mmse_train = torch.randn(100)  # Example ground truth MMSE scores
y_true_bins_train = torch.randint(0, 10, (100,))  # Example ground truth binned MMSE interval labels

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_true_mmse_train_tensor = torch.tensor(y_true_mmse_train)
y_true_bins_train_tensor = torch.tensor(y_true_bins_train)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_true_mmse_train_tensor, y_true_bins_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Assuming we have some fake validation data
X_val = torch.randn(50, 10)  # Example input data with 50 samples and 10 features
y_true_mmse_val = torch.randn(50)  # Example ground truth MMSE scores for validation
y_true_bins_val = torch.randint(0, 10, (50,))  # Example ground truth binned MMSE interval labels for validation

# Convert validation data to PyTorch tensors
X_val_tensor = torch.tensor(X_val)
y_true_mmse_val_tensor = torch.tensor(y_true_mmse_val)
y_true_bins_val_tensor = torch.tensor(y_true_bins_val)

# Create a TensorDataset for validation
val_dataset = TensorDataset(X_val_tensor, y_true_mmse_val_tensor, y_true_bins_val_tensor)
# Create DataLoader for validation
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Let's print the first batch to see how it looks
for batch_idx, (inputs, mmse_labels, bins_labels) in enumerate(train_loader):
    print("Batch", batch_idx)
    print("Input Shape:", inputs.shape)
    print("MMSE Labels Shape:", mmse_labels.shape)
    print("Bins Labels Shape:", bins_labels.shape)
    break  # Print only the first batch for brevity

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_mmse_rmse = 0.0
    total_accuracy = 0.0

    for inputs, mmse_labels, bins_labels in train_loader:
        optimizer.zero_grad()
        mmse_preds, bins_logits, _ = model(inputs)
        loss = loss_fn(mmse_labels, bins_labels, mmse_preds, bins_logits)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mmse_rmse += rmse(mmse_preds, mmse_labels).item()

        # Convert logits to categorical predictions
        _, predicted_bins = torch.max(bins_logits, 1)
        total_accuracy += torch.sum(predicted_bins == bins_labels).item() / len(bins_labels)

    avg_train_loss = total_loss / len(train_loader)
    avg_train_mmse_rmse = total_mmse_rmse / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)
    print(avg_train_loss, avg_train_mmse_rmse, avg_train_accuracy)

# Evaluation phase
model.eval()
total_val_loss = 0.0
total_val_mmse_rmse = 0.0
total_val_accuracy = 0.0

with torch.no_grad():
    for inputs, mmse_labels, bins_labels in val_loader:
        mmse_preds, bins_logits, _ = model(inputs)
        loss = loss_fn(mmse_labels, bins_labels, mmse_preds, bins_logits)
        total_val_loss += loss.item()
        total_val_mmse_rmse += rmse(mmse_preds, mmse_labels).item()

        # Convert logits to categorical predictions
        _, predicted_bins = torch.max(bins_logits, 1)
        total_val_accuracy += torch.sum(predicted_bins == bins_labels).item() / len(bins_labels)

        # Calculate average validation metrics
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_mmse_rmse = total_val_mmse_rmse / len(val_loader)
    avg_val_accuracy = total_val_accuracy / len(val_loader)

    print(
        f"Validation Loss: {avg_val_loss}, Validation MMSE RMSE: {avg_val_mmse_rmse}, Validation Accuracy: {avg_val_accuracy}")
