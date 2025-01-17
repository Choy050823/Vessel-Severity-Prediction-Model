import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """
    Train the final model using the best hyperparameters.
    """
    # Convert labels to 0-indexed format
    y_train = y_train - 1  # Subtract 1 to make labels 0-indexed
    y_test = y_test - 1    # Subtract 1 to make labels 0-indexed
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define the final model
    class FinalModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(FinalModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, best_params['hidden_dim'])
            self.fc2 = nn.Linear(best_params['hidden_dim'], best_params['hidden_dim'] // 2)
            self.fc3 = nn.Linear(best_params['hidden_dim'] // 2, num_classes)
            self.dropout = nn.Dropout(best_params['dropout_rate'])
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Initialize the final model
    final_model = FinalModel(X_train.shape[1], len(np.unique(y_train)))
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(50):  # Train for more epochs
        final_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        final_model.eval()
        with torch.no_grad():
            outputs = final_model(X_test_tensor)
            _, y_pred = torch.max(outputs, dim=1)
            accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Test Accuracy: {accuracy}")

    # Save the final model
    torch.save(final_model.state_dict(), 'final_model.pth')

    return final_model